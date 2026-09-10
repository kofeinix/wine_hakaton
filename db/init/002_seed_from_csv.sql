CREATE TEMP TABLE staging_wineries (
    winery_id text,
    name text,
    address text,
    description text,
    vineyard_area text,
    region text,
    locality text,
    climate text
);

COPY staging_wineries
FROM '/seed/export/wineries.csv'
WITH (FORMAT csv, HEADER true);

INSERT INTO wineries (
    winery_id,
    name,
    address,
    description,
    vineyard_area,
    region,
    locality,
    climate
)
SELECT
    winery_id::uuid,
    name,
    NULLIF(address, ''),
    NULLIF(description, ''),
    NULLIF(vineyard_area, ''),
    NULLIF(region, ''),
    NULLIF(locality, ''),
    NULLIF(climate, '')
FROM staging_wineries
ON CONFLICT (winery_id) DO UPDATE SET
    name = EXCLUDED.name,
    address = EXCLUDED.address,
    description = EXCLUDED.description,
    vineyard_area = EXCLUDED.vineyard_area,
    region = EXCLUDED.region,
    locality = EXCLUDED.locality,
    climate = EXCLUDED.climate;

CREATE TEMP TABLE staging_wines (
    wine_id text,
    winery_id text,
    name text,
    producer text,
    rating text,
    color text,
    wine_type text,
    region text,
    grape_variety text,
    shade text,
    description text,
    serving_temperature text,
    alcohol text,
    food_pairing text,
    url text,
    photo_url text
);

COPY staging_wines
FROM '/seed/export/wines.csv'
WITH (FORMAT csv, HEADER true);

INSERT INTO wines (
    wine_id,
    winery_id,
    name,
    producer,
    rating,
    color,
    wine_type,
    region,
    grape_varieties,
    shade,
    description,
    serving_temperature,
    alcohol,
    food_pairings,
    minio_photo_path
)
SELECT
    wine_id::uuid,
    NULLIF(winery_id, '')::uuid,
    name,
    NULLIF(producer, ''),
    NULLIF(rating, '')::numeric(3, 2),
    NULLIF(color, ''),
    NULLIF(wine_type, ''),
    NULLIF(region, ''),
    COALESCE(regexp_split_to_array(NULLIF(grape_variety, ''), '\s*,\s*'), ARRAY[]::text[]),
    NULLIF(shade, ''),
    NULLIF(description, ''),
    NULLIF(serving_temperature, ''),
    NULLIF(alcohol, ''),
    COALESCE(regexp_split_to_array(NULLIF(food_pairing, ''), '\s*,\s*'), ARRAY[]::text[]),
    'wine/' || wine_id
FROM staging_wines
ON CONFLICT (wine_id) DO UPDATE SET
    winery_id = EXCLUDED.winery_id,
    name = EXCLUDED.name,
    producer = EXCLUDED.producer,
    rating = EXCLUDED.rating,
    color = EXCLUDED.color,
    wine_type = EXCLUDED.wine_type,
    region = EXCLUDED.region,
    grape_varieties = EXCLUDED.grape_varieties,
    shade = EXCLUDED.shade,
    description = EXCLUDED.description,
    serving_temperature = EXCLUDED.serving_temperature,
    alcohol = EXCLUDED.alcohol,
    food_pairings = EXCLUDED.food_pairings,
    minio_photo_path = EXCLUDED.minio_photo_path;
