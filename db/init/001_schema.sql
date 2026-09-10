CREATE TABLE IF NOT EXISTS wineries (
    winery_id uuid PRIMARY KEY,
    name text NOT NULL,
    address text,
    description text,
    vineyard_area text,
    region text,
    locality text,
    climate text
);

CREATE INDEX IF NOT EXISTS ix_wineries_name ON wineries (name);
CREATE INDEX IF NOT EXISTS ix_wineries_region ON wineries (region);

CREATE TABLE IF NOT EXISTS wines (
    wine_id uuid PRIMARY KEY,
    winery_id uuid REFERENCES wineries (winery_id) ON DELETE SET NULL,
    name text NOT NULL,
    producer text,
    rating numeric(3, 2),
    color text,
    wine_type text,
    region text,
    grape_varieties text[] NOT NULL DEFAULT ARRAY[]::text[],
    shade text,
    description text,
    serving_temperature text,
    alcohol text,
    food_pairings text[] NOT NULL DEFAULT ARRAY[]::text[],
    minio_photo_path text,
    CONSTRAINT ck_wines_rating_range CHECK (rating IS NULL OR rating BETWEEN 0 AND 5)
);

CREATE INDEX IF NOT EXISTS ix_wines_winery_id ON wines (winery_id);
CREATE INDEX IF NOT EXISTS ix_wines_name ON wines (name);
CREATE INDEX IF NOT EXISTS ix_wines_producer ON wines (producer);
CREATE INDEX IF NOT EXISTS ix_wines_wine_type ON wines (wine_type);
CREATE INDEX IF NOT EXISTS ix_wines_region ON wines (region);
CREATE INDEX IF NOT EXISTS ix_wines_name_producer ON wines (name, producer);
