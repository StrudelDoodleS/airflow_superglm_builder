IF OBJECT_ID('pricing.PRICING_MODEL_VERSION_RESERVATION', 'U') IS NULL
BEGIN
    CREATE TABLE pricing.PRICING_MODEL_VERSION_RESERVATION (
        model_id BIGINT NOT NULL,
        export_id NVARCHAR(128) NOT NULL,
        model_version NVARCHAR(64) NOT NULL,
        reserved_ts DATETIME2(3) NOT NULL
            CONSTRAINT DF_MODEL_VERSION_RESERVATION_TS DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_MODEL_VERSION_RESERVATION
            PRIMARY KEY (model_id, export_id),
        CONSTRAINT UQ_MODEL_VERSION_RESERVATION_VERSION
            UNIQUE (model_id, model_version),
        CONSTRAINT FK_MODEL_VERSION_RESERVATION_MODEL
            FOREIGN KEY (model_id)
            REFERENCES pricing.PRICING_MODEL(model_id)
    );
END;
GO

;WITH ranked_packages AS (
    SELECT
        rp.model_id,
        rp.source_export_id AS export_id,
        rp.model_version,
        ROW_NUMBER() OVER (
            PARTITION BY rp.model_id, rp.model_version
            ORDER BY rp.rate_package_id
        ) AS version_rank
    FROM pricing.PRICING_RATE_PACKAGE AS rp
    WHERE rp.parent_rate_package_id IS NULL
      AND rp.model_id IS NOT NULL
      AND rp.source_export_id IS NOT NULL
      AND rp.model_version IS NOT NULL
)
INSERT INTO pricing.PRICING_MODEL_VERSION_RESERVATION (
    model_id,
    export_id,
    model_version
)
SELECT
    ranked.model_id,
    ranked.export_id,
    ranked.model_version
FROM ranked_packages AS ranked
WHERE ranked.version_rank = 1
  AND NOT EXISTS (
      SELECT 1
      FROM pricing.PRICING_MODEL_VERSION_RESERVATION AS existing
      WHERE existing.model_id = ranked.model_id
        AND (
            existing.export_id = ranked.export_id
            OR existing.model_version = ranked.model_version
        )
  );
GO
