use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::{Initializer, LinearConfig}, optim::{AdamConfig, GradientsParams},
    tensor::{Int, TensorData}};
use std::collections::{HashMap, HashSet};
type State = (usize, usize, usize);
#[derive(Clone, Debug)]
struct Item { position: (usize, usize), name: &'static str, value: f32 }
#[derive(Clone, Debug)]
struct World { rows: usize, cols: usize, walls: HashSet<(usize, usize)>, items: Vec<Item> }
impl World {
    fn new(rows: usize, cols: usize, walls: &[(usize, usize)], mut items: Vec<Item>) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows.saturating_mul(cols) < 2 {
            return Err("Поле должно содержать хотя бы две клетки".into());
        }
        let inside = |p: (usize, usize)| p.0 < rows && p.1 < cols;
        let walls: HashSet<_> = walls.iter().copied().collect();
        let goal = (rows - 1, cols - 1);
        if walls.iter().any(|&p| !inside(p)) || items.iter().any(|i| !inside(i.position)) {
            return Err("Стена или предмет находится за границей поля".into());
        }
        if walls.contains(&(0, 0)) || walls.contains(&goal) {
            return Err("Старт и выход должны быть свободны".into());
        }
        let mut occupied = HashSet::new();
        for item in &items {
            if walls.contains(&item.position) || item.position == (0, 0) || item.position == goal
                || !occupied.insert(item.position) || !item.value.is_finite() || item.value.abs() >= 1e6 {
                return Err("Проверь положение и стоимость предметов".into());
            }
        }
        if items.len() > 6 || (rows.saturating_mul(cols) - walls.len() - 1)
            .saturating_mul(1_usize << items.len()) > 2048 {
            return Err("Для учебного опыта уменьши поле или число предметов".into());
        }
        items.sort_by_key(|item| item.position);
        let world = Self { rows, cols, walls, items };
        let mut reached = HashSet::from([(0, 0)]);
        let mut queue = vec![(0, 0)];
        let mut index = 0;
        while index < queue.len() {
            let (r, c) = queue[index];
            index += 1;
            if (r, c) == goal { continue; }
            for (dr, dc) in [(-1_isize, 0_isize), (1, 0), (0, -1), (0, 1)] {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize { continue; }
                let next = (nr as usize, nc as usize);
                if !world.walls.contains(&next) && reached.insert(next) { queue.push(next); }
            }
        }
        if !reached.contains(&goal) || world.items.iter().any(|i| !reached.contains(&i.position)) {
            return Err("Из старта должны быть достижимы выход и все предметы".into());
        }
        Ok(world)
    }
    fn step(&self, state: State, action: usize) -> (State, f32, bool) {
        assert!(action < 4, "Действия: 0 вверх, 1 вниз, 2 влево, 3 вправо");
        let (row, col, mut collected) = state;
        let goal = (self.rows - 1, self.cols - 1);
        if (row, col) == goal { return (state, 0.0, true); }
        let (dr, dc) = [(-1_isize, 0_isize), (1, 0), (0, -1), (0, 1)][action];
        let nr = row as isize + dr;
        let nc = col as isize + dc;
        if nr < 0 || nc < 0 || nr >= self.rows as isize || nc >= self.cols as isize {
            return (state, -0.1, false);
        }
        let next = (nr as usize, nc as usize);
        if self.walls.contains(&next) { return (state, -0.1, false); }
        let mut reward = -0.04;
        for (index, item) in self.items.iter().enumerate() {
            let bit = 1 << index;
            if next == item.position && collected & bit == 0 {
                reward += item.value;
                collected |= bit;
            }
        }
        let done = next == goal;
        if done { reward += 2.0; }
        ((next.0, next.1, collected), reward, done)
    }
}
fn train(world: &World, device: &Device) -> (HashMap<State, usize>, f32) {
    let mut states = Vec::new();
    for mask in 0..(1 << world.items.len()) {
        for r in 0..world.rows { for c in 0..world.cols {
            if !world.walls.contains(&(r, c)) && (r, c) != (world.rows - 1, world.cols - 1) {
                states.push((r, c, mask));
            }
        }}
    }
    let lookup: HashMap<_, _> = states.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let count = states.len();
    let mut next_ids = Vec::new();
    let mut rewards = Vec::new();
    let mut alive = Vec::new();
    for &state in &states { for action in 0..4 {
        let (next, reward, done) = world.step(state, action);
        next_ids.push(*lookup.get(&next).unwrap_or(&0) as i64);
        rewards.push(reward);
        alive.push(if done { 0.0_f32 } else { 1.0 });
    }}
    let next_ids = Tensor::<1, Int>::from_data(TensorData::new(next_ids, [count * 4]), device);
    let rewards = Tensor::<2>::from_data(TensorData::new(rewards, [count, 4]), device);
    let alive = Tensor::<2>::from_data(TensorData::new(alive, [count, 4]), device);

    // Один вход на каждое состояние, четыре выхода — оценки действий Q(s, a).
    let features = Tensor::<2>::eye(count, device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = LinearConfig::new(count, 4).with_bias(false)
        .with_initializer(Initializer::Zeros).init(device);

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut optimizer = AdamConfig::new().init();
    let mut target = rewards.clone();
    let mut last_loss = 0.0;
    for update in 0..1400 {
        if update % 20 == 0 {
            // Цель заморожена на 20 обновлений; через неё градиенты не проходят.
            let future = model.forward(features.clone()).detach().max_dim(1)

                // Меняем форму тензора, сохраняя число и порядок элементов.
                .squeeze::<1>().select(0, next_ids.clone()).reshape([count, 4]);
            target = rewards.clone() + future * alive.clone() * 0.95;
        }
        let loss = (model.forward(features.clone()) - target.clone()).powf_scalar(2.0).mean();

        // Читаем единственное значение тензора как обычное число Rust.
        if update == 1399 { last_loss = loss.clone().into_scalar::<f32>(); }

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model = optimizer.step(0.06, model, grads);
    }
    let scores = model.forward(features).into_data().try_to_vec::<f32>().unwrap();
    let policy = states.into_iter().enumerate().map(|(i, state)| {
        let mut best = 0;
        for action in 1..4 { if scores[i * 4 + action] > scores[i * 4 + best] { best = action; } }
        (state, best)
    }).collect();
    (policy, last_loss)
}
struct Evaluation { route: Vec<State>, actions: Vec<usize>, total: f32, discounted: f32, done: bool }
fn rollout(world: &World, mut choose: impl FnMut(State) -> usize) -> Evaluation {
    let mut state = (0, 0, 0);
    let mut result = Evaluation { route: vec![state], actions: vec![], total: 0.0, discounted: 0.0, done: false };
    let mut discount = 1.0;
    for _ in 0..world.rows * world.cols * 8 {
        let action = choose(state);
        let (next, reward, done) = world.step(state, action);
        state = next;
        result.route.push(state);
        result.actions.push(action);
        result.total += reward;
        result.discounted += discount * reward;
        result.done = done;
        discount *= 0.95;
        if done { break; }
    }
    result
}
fn show(world: &World, result: &Evaluation) {
    let visited: HashSet<_> = result.route.iter().map(|s| (s.0, s.1)).collect();
    println!("A — старт; G — выход; # — стена; * — маршрут; цифры — предметы.");
    for r in 0..world.rows {
        for c in 0..world.cols {
            let p = (r, c);
            let symbol = if p == (0, 0) { 'A' }
                else if p == (world.rows - 1, world.cols - 1) { 'G' }
                else if world.walls.contains(&p) { '#' }
                else if let Some(i) = world.items.iter().position(|item| item.position == p) {
                    char::from_digit(i as u32 + 1, 10).unwrap()
                } else if visited.contains(&p) { '*' } else { '.' };
            print!("{symbol} ");
        }
        println!();
    }
    let mask = result.route.last().unwrap().2;
    for (i, item) in world.items.iter().enumerate() {
        println!("{}: {}, стоимость {}, клетка {:?}, собран: {}",
            i + 1, item.name, item.value, item.position, mask & (1 << i) != 0);
    }
    let arrows: Vec<_> = result.actions.iter().map(|&a| ["↑", "↓", "←", "→"][a]).collect();
    println!("Действия: {}", arrows.join(" "));
    println!("Маршрут: {:?}", result.route.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>());
    println!("Выход достигнут: {} | шагов: {}", result.done, result.actions.len());
    println!("Сумма наград: {:.3} | с учётом скидки: {:.3}", result.total, result.discounted);
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let world = demo_world();
    println!("Поле {}×{}; 0 ↑, 1 ↓, 2 ←, 3 →; скидка будущей награды 0.95.", world.rows, world.cols);
    let mut random = 7_u64;
    let mut successes = 0;
    let mut baseline = 0.0;
    for _ in 0..100 {
        let result = rollout(&world, |_| {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((random >> 32) % 4) as usize
        });
        successes += usize::from(result.done);
        baseline += result.discounted;
    }
    println!("Случайный агент: успехов {successes} / 100; средняя награда со скидкой {:.3}", baseline / 100.0);
    let (policy, loss) = train(&world, &device);
    let result = rollout(&world, |state| policy[&state]);
    println!("Ошибка обучения: {loss:.6}");
    show(&world, &result);
    assert!(result.done, "Агент не дошёл до выхода; проверь обучение и карту");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn movement_rewards_and_terminal_state() {
        let world = World::new(3, 5, &[(1, 3)], vec![
            Item { position: (1, 2), name: "Монета", value: 2.0 },
            Item { position: (2, 1), name: "Алмаз", value: 7.0 },
        ]).unwrap();
        for (action, expected) in [(0, 1), (2, 1), (1, 0), (1, 2)].iter().enumerate() {
            let (next, _, done) = world.step((1, 1, 0), action);
            assert_eq!((next.0, next.1), *expected);
            assert!(!done);
        }
        assert_eq!(world.step((0, 0, 0), 0), ((0, 0, 0), -0.1, false));
        assert_eq!(world.step((0, 0, 0), 2), ((0, 0, 0), -0.1, false));
        assert_eq!(world.step((2, 0, 0), 1), ((2, 0, 0), -0.1, false));
        assert_eq!(world.step((0, 4, 0), 3), ((0, 4, 0), -0.1, false));
        assert_eq!(world.step((1, 2, 0), 3), ((1, 2, 0), -0.1, false));
        let (state, reward, _) = world.step((1, 1, 0), 3);
        assert_eq!(state, (1, 2, 1));
        assert!((reward - 1.96).abs() < 1e-5);
        assert_eq!(world.step((1, 1, 1), 3), ((1, 2, 1), -0.04, false));
        let (state, reward, done) = world.step((2, 3, 3), 3);
        assert_eq!(state, (2, 4, 3));
        assert!((reward - 1.96).abs() < 1e-5 && done);
        assert_eq!(world.step(state, 2), (state, 0.0, true));
    }
    #[test]
    fn invalid_worlds_are_rejected() {
        assert!(World::new(0, 5, &[], vec![]).is_err());
        assert!(World::new(3, 5, &[(0, 0)], vec![]).is_err());
        assert!(World::new(3, 5, &[(2, 4)], vec![]).is_err());
        assert!(World::new(3, 5, &[(3, 0)], vec![]).is_err());
        assert!(World::new(3, 4, &[(0, 3), (1, 2)], vec![
            Item { position: (1, 3), name: "X", value: 1.0 },
        ]).is_err());
        assert!(World::new(3, 5, &[(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)], vec![]).is_err());
    }
}

// Для этой расстановки нужны N >= 4 и M >= 5.
const N: usize = 4;
const M: usize = 5;
fn demo_world() -> World {
    assert!(N >= 4 && M >= 5, "Для готовой карты нужны N >= 4 и M >= 5");
    World::new(N, M, &[(1, 1), (1, 2), (2, 3)], vec![
        Item { position: (0, M - 1), name: "Монета", value: 1.0 },
        Item { position: (2, 1), name: "Алмаз", value: 5.0 },
        Item { position: (N - 1, 0), name: "Сундук", value: 9.0 },
    ]).expect("Проверь стены и предметы")
}
