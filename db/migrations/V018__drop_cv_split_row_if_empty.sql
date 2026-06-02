IF OBJECT_ID('mlops.CV_SPLIT_ROW', 'U') IS NOT NULL
BEGIN
    IF EXISTS (SELECT 1 FROM mlops.CV_SPLIT_ROW)
        THROW 51002, 'Cannot drop mlops.CV_SPLIT_ROW because it contains row-level CV split assignments. Move split assignments to npz artifacts before rerunning migrations.', 1;

    DROP TABLE mlops.CV_SPLIT_ROW;
END;
GO
