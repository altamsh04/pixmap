import Database from 'better-sqlite3';
import { ImageRecord } from './types.js';

export class MetadataDb {
  private db: Database.Database;

  constructor(path: string) {
    this.db = new Database(path);
    this.db.pragma('journal_mode = WAL');
    this.migrateSchema();
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS images (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        path     TEXT NOT NULL UNIQUE,
        indexed  INTEGER NOT NULL DEFAULT 0,
        created  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
      )
    `);
  }

  upsert(path: string): { id: number; indexed: boolean } {
    const existing = this.findByPath(path);
    if (existing) {
      return { id: existing.id, indexed: existing.indexed === 1 };
    }

    const result = this.db.prepare('INSERT INTO images (path, indexed) VALUES (?, 0)').run(path);

    return { id: Number(result.lastInsertRowid), indexed: false };
  }

  markIndexed(id: number): void {
    this.db.prepare('UPDATE images SET indexed = 1 WHERE id = ?').run(id);
  }

  get(id: number): ImageRecord | undefined {
    return this.db.prepare('SELECT * FROM images WHERE id = ?').get(id) as ImageRecord | undefined;
  }

  list(limit = 200): ImageRecord[] {
    return this.db
      .prepare('SELECT * FROM images WHERE indexed = 1 ORDER BY created DESC, id DESC LIMIT ?')
      .all(limit) as ImageRecord[];
  }

  findByPath(path: string): ImageRecord | undefined {
    return this.db.prepare('SELECT * FROM images WHERE path = ?').get(path) as ImageRecord | undefined;
  }

  private migrateSchema(): void {
    const exists = this.db
      .prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'images'"
      )
      .get() as { name: string } | undefined;

    if (!exists) {
      return;
    }

    const columns = this.db
      .prepare("PRAGMA table_info('images')")
      .all() as Array<{ name: string }>;
    const allowed = new Set(['id', 'path', 'indexed', 'created']);
    const isCurrentSchema =
      columns.length === allowed.size && columns.every((c) => allowed.has(c.name));

    if (isCurrentSchema) {
      return;
    }

    this.db.exec(`
      BEGIN;
      CREATE TABLE images_new (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        path     TEXT NOT NULL UNIQUE,
        indexed  INTEGER NOT NULL DEFAULT 0,
        created  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
      );
      INSERT INTO images_new (id, path, indexed, created)
      SELECT id, path, indexed, created FROM images;
      DROP TABLE images;
      ALTER TABLE images_new RENAME TO images;
      COMMIT;
    `);
  }
}
