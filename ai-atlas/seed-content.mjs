import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';
import { database, seedRecords, seedDatabase } from './content-db.mjs';
import { editorialSchema } from './editorial-db.mjs';
import { agentSchema } from './agent-db.mjs';
import { searchSchema } from './search-db.mjs';
import { editSchema } from './content-edit-api.mjs';

// Only the production build seeds the dedicated content database. Existing rows win.
if (process.env.DATABASE_URL && (!process.env.VERCEL_ENV || process.env.VERCEL_ENV === 'production')) {
  const scope = { window: {} };
  for (const file of ['map-data.js', 'lesson-data.js']) {
    runInNewContext(await readFile(new URL(file, import.meta.url), 'utf8'), scope);
  }
  const records = seedRecords(scope.window.ATLAS_DATA, scope.window.ATLAS_LESSONS);
  const sql = await database();
  await seedDatabase(sql, records);
  await sql.transaction(editorialSchema.map(statement => sql.query(statement)));
  await sql.transaction(editSchema.map(statement => sql.query(statement)));
  await sql.transaction(agentSchema.map(statement => sql.query(statement)));
  await sql.transaction(searchSchema.map(statement => sql.query(statement)));
  console.log(`Content database ready: ${records.length} records; existing edits preserved.`);
}
