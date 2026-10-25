use burn::{
    module::Module,
    nn::{conv::Conv2dConfig, PaddingConfig2d},
    tensor::{Device, Distribution, Tensor},
};

fn main() {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);

    // 1. Как torch.randn(1, 16, 24, 24): случайный вход, а не датасет.
    // Это симуляция. Порядок осей NCHW: пакет, каналы, высота, ширина.
    // В этом примере используются веса FP32: 4 байта на число.
    let x = Tensor::<4>::random(
        [1, 16, 24, 24],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // 2. Аналог nn.Conv2d(16, 32, 3, padding=1, bias=False).
    // При фильтре 3×3 и шаге 1 Same добавляет по одному нулю с краёв.
    let regular = Conv2dConfig::new([16, 32], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(&device);

    // 3. Аналог двух слоёв в nn.Sequential из Python.
    // Depthwise: отдельный фильтр 3×3 для каждого из 16 каналов.
    let depthwise = Conv2dConfig::new([16, 16], [3, 3])
        .with_groups(16)
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(&device);

    // Pointwise: смешиваем 16 каналов в 32 свёрткой 1×1.
    // Между свёртками нет активации — точно как в исходном Python-коде.
    let pointwise = Conv2dConfig::new([16, 32], [1, 1])
        .with_bias(false)
        .init(&device);

    // 4. Обе ветви получают один и тот же вход.
    // clone() сохраняет возможность передать x и второй ветви;
    // новый случайный вход здесь не создаётся.
    let regular_output = regular.forward(x.clone());
    let depthwise_output = depthwise.forward(x);
    let compact_output = pointwise.forward(depthwise_output);

    // 5. Аналог sum(p.numel() for p in model.parameters()).
    let regular_count = regular.num_params();
    let compact_count = depthwise.num_params() + pointwise.num_params();
    let bytes_per_weight = std::mem::size_of::<f32>();

    for (name, count, shape) in [
        ("Обычная", regular_count, regular_output.dims()),
        ("Компактная", compact_count, compact_output.dims()),
    ] {
        println!(
            "{name} параметров: {count}, байт весов: {}, размер ответа: {shape:?}",
            count * bytes_per_weight,
        );
    }

    // Проверяем не только совпадение форм, но и ожидаемые размеры.
    assert_eq!(regular_count, 32 * 16 * 3 * 3);
    assert_eq!(compact_count, 16 * 3 * 3 + 32 * 16);
    assert_eq!(regular_count * bytes_per_weight, 18_432);
    assert_eq!(compact_count * bytes_per_weight, 2_624);
    assert_eq!(regular_output.dims(), [1, 32, 24, 24]);
    assert_eq!(compact_output.dims(), [1, 32, 24, 24]);
}
