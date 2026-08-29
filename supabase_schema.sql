-- Indian Pharma & Generic Drug Substitute Intelligence Engine
-- Apply in the Supabase SQL editor or through the project's migration workflow.

create extension if not exists vector with schema extensions;

create table if not exists public.medicines (
    id uuid primary key default gen_random_uuid(),
    source_id text not null unique,
    drug_code text,
    unit_size text,
    therapeutic_group text,
    generic_name text not null,
    brand_name text not null,
    manufacturer text not null,
    strength text not null,
    dosage_form text not null,
    jan_aushadhi_mrp numeric(12, 2),
    brand_mrp numeric(12, 2),
    contraindications text not null default '',
    source_url text not null,
    source_date date,
    canonical_text text not null,
    search_vector tsvector generated always as (
        to_tsvector(
            'simple'::regconfig,
            coalesce(generic_name, '') || ' ' ||
            coalesce(brand_name, '') || ' ' ||
            coalesce(manufacturer, '') || ' ' ||
            coalesce(strength, '') || ' ' ||
            coalesce(dosage_form, '') || ' ' ||
            coalesce(canonical_text, '')
        )
    ) stored,
    embedding vector(384),
    embedding_model text not null default 'BAAI/bge-small-en-v1.5',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint medicines_positive_prices check (
        jan_aushadhi_mrp is null or jan_aushadhi_mrp >= 0
    ),
    constraint medicines_positive_brand_price check (brand_mrp is null or brand_mrp >= 0)
);

create index if not exists medicines_search_vector_gin_idx
    on public.medicines using gin (search_vector);
create index if not exists medicines_generic_name_idx
    on public.medicines (lower(generic_name));
create index if not exists medicines_brand_name_idx
    on public.medicines (lower(brand_name));
create index if not exists medicines_embedding_hnsw_idx
    on public.medicines using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

create table if not exists public.medicine_chunks (
    id uuid primary key default gen_random_uuid(),
    source_id text not null unique,
    source_path text not null,
    source_url text not null,
    page_number integer not null check (page_number > 0),
    chunk_index integer not null check (chunk_index >= 0),
    content text not null,
    search_vector tsvector generated always as (
        to_tsvector('simple'::regconfig, coalesce(content, ''))
    ) stored,
    embedding vector(384) not null,
    embedding_model text not null default 'BAAI/bge-small-en-v1.5',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists medicine_chunks_search_vector_gin_idx
    on public.medicine_chunks using gin (search_vector);
create index if not exists medicine_chunks_embedding_hnsw_idx
    on public.medicine_chunks using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

create or replace function public.match_medicines_dense(
    query_embedding vector(384),
    match_count integer default 25
)
returns table (
    id uuid,
    source_id text,
    generic_name text,
    brand_name text,
    manufacturer text,
    strength text,
    dosage_form text,
    jan_aushadhi_mrp numeric,
    brand_mrp numeric,
    contraindications text,
    source_url text,
    source_date date,
    canonical_text text,
    similarity real
)
language sql
stable
set search_path = public, extensions
as $$
    select
        m.id, m.source_id, m.generic_name, m.brand_name, m.manufacturer,
        m.strength, m.dosage_form, m.jan_aushadhi_mrp, m.brand_mrp,
        m.contraindications, m.source_url, m.source_date, m.canonical_text,
        (1 - (m.embedding <=> query_embedding))::real as similarity
    from public.medicines as m
    where m.embedding is not null
    order by m.embedding <=> query_embedding
    limit least(greatest(match_count, 1), 25);
$$;

create or replace function public.match_medicines_sparse(
    query_text text,
    match_count integer default 25
)
returns table (
    id uuid,
    source_id text,
    generic_name text,
    brand_name text,
    manufacturer text,
    strength text,
    dosage_form text,
    jan_aushadhi_mrp numeric,
    brand_mrp numeric,
    contraindications text,
    source_url text,
    source_date date,
    canonical_text text,
    lexical_score real
)
language sql
stable
set search_path = public, extensions
as $$
    select
        m.id, m.source_id, m.generic_name, m.brand_name, m.manufacturer,
        m.strength, m.dosage_form, m.jan_aushadhi_mrp, m.brand_mrp,
        m.contraindications, m.source_url, m.source_date, m.canonical_text,
        ts_rank_cd(
            m.search_vector,
            websearch_to_tsquery('simple', regexp_replace(query_text, '\s+', ' OR ', 'g'))
        )::real
    from public.medicines as m
    where m.search_vector @@ websearch_to_tsquery(
        'simple', regexp_replace(query_text, '\s+', ' OR ', 'g')
    )
    order by ts_rank_cd(
        m.search_vector,
        websearch_to_tsquery('simple', regexp_replace(query_text, '\s+', ' OR ', 'g'))
    ) desc
    limit least(greatest(match_count, 1), 25);
$$;

create or replace function public.match_medicine_chunks_dense(
    query_embedding vector(384),
    match_count integer default 25
)
returns table (
    source_id text,
    canonical_text text,
    source_path text,
    source_url text,
    page_number integer,
    chunk_index integer,
    similarity real
)
language sql
stable
set search_path = public, extensions
as $$
    select c.source_id, c.content, c.source_path, c.source_url,
        c.page_number, c.chunk_index,
        (1 - (c.embedding <=> query_embedding))::real
    from public.medicine_chunks as c
    order by c.embedding <=> query_embedding
    limit least(greatest(match_count, 1), 25);
$$;

create or replace function public.match_medicine_chunks_sparse(
    query_text text,
    match_count integer default 25
)
returns table (
    source_id text,
    canonical_text text,
    source_path text,
    source_url text,
    page_number integer,
    chunk_index integer,
    lexical_score real
)
language sql
stable
set search_path = public, extensions
as $$
    select c.source_id, c.content, c.source_path, c.source_url,
        c.page_number, c.chunk_index,
        ts_rank_cd(
            c.search_vector,
            websearch_to_tsquery('simple', regexp_replace(query_text, '\s+', ' OR ', 'g'))
        )::real
    from public.medicine_chunks as c
    where c.search_vector @@ websearch_to_tsquery(
        'simple', regexp_replace(query_text, '\s+', ' OR ', 'g')
    )
    order by ts_rank_cd(
        c.search_vector,
        websearch_to_tsquery('simple', regexp_replace(query_text, '\s+', ' OR ', 'g'))
    ) desc
    limit least(greatest(match_count, 1), 25);
$$;

alter table public.medicines enable row level security;
alter table public.medicine_chunks enable row level security;
drop policy if exists "Public can read medicine intelligence" on public.medicines;
create policy "Public can read medicine intelligence"
    on public.medicines for select
to anon, authenticated
using (true);

revoke insert, update, delete on public.medicines from anon, authenticated;
grant select on public.medicines to anon, authenticated;
drop policy if exists "Public can read medicine chunks" on public.medicine_chunks;
create policy "Public can read medicine chunks"
    on public.medicine_chunks for select
to anon, authenticated
using (true);
revoke insert, update, delete on public.medicine_chunks from anon, authenticated;
grant select on public.medicine_chunks to anon, authenticated;
revoke all on function public.match_medicines_dense(vector, integer) from public;
revoke all on function public.match_medicines_sparse(text, integer) from public;
grant execute on function public.match_medicines_dense(vector, integer) to anon, authenticated;
grant execute on function public.match_medicines_sparse(text, integer) to anon, authenticated;
revoke all on function public.match_medicine_chunks_dense(vector, integer) from public;
revoke all on function public.match_medicine_chunks_sparse(text, integer) from public;
grant execute on function public.match_medicine_chunks_dense(vector, integer) to anon, authenticated;
grant execute on function public.match_medicine_chunks_sparse(text, integer) to anon, authenticated;
