SELECT
    f.JobNumber,
    f.WarehouseId,
    f.DivisionName,
    f.State,
    f.WarehouseName,
    f.StreetAddress,

    -- ── Actual stage complete dates (from BuildPro tasks) ──
    f.ReleasedCompleteDate,
    f.FoundationCompleteDate    AS ActualFoundationDate,
    f.FrameCompleteDate         AS ActualFrameDate,
    f.CorniceCompleteDate       AS ActualCorniceDate,
    f.MechanicalsCompleteDate   AS ActualMechanicalsDate,
    f.SheetrockCompleteDate     AS ActualSheetrockDate,
    f.TrimCompleteDate          AS ActualTrimDate,
    f.InteriorCompleteDate      AS ActualInteriorDate,
    f.FinalCompleteDate         AS ActualFinalDate,
    f.CloseDate                 AS ActualCloseDate,

    -- ── 4-month avg model predictions (old model) ──────────
    f.ProjectedFoundationDate   AS FourMonthAvgFoundationDate,
    f.ProjectedFrameDate        AS FourMonthAvgFrameDate,
    f.ProjectedCorniceDate      AS FourMonthAvgCorniceDate,
    f.ProjectedMechanicalsDate  AS FourMonthAvgMechanicalsDate,
    f.ProjectedSheetrockDate    AS FourMonthAvgSheetrockDate,
    f.ProjectedTrimDate         AS FourMonthAvgTrimDate,
    f.ProjectedInteriorDate     AS FourMonthAvgInteriorDate,
    f.ProjectedFinalDate        AS FourMonthAvgFinalDate,
    f.ProjectedCloseDate        AS FourMonthAvgCloseDate,

    -- ── Actual durations (training targets) ───────────────
    f.FoundationDuration,
    f.FrameDuration,
    f.CorniceDuration,
    f.MechanicalsDuration,
    f.SheetrockDuration,
    f.TrimDuration,
    f.InteriorDuration,
    f.FinalDuration,
    f.CloseDuration,

    -- ── Division / project rolling averages (features) ────
    f.FoundationDurationAvgProj,  f.FoundationDurationAvgDiv,
    f.FrameDurationAvgProj,       f.FrameDurationAvgDiv,
    f.CorniceDurationAvgProj,     f.CorniceDurationAvgDiv,
    f.MechanicalsDurationAvgProj, f.MechanicalsDurationAvgDiv,
    f.SheetrockDurationAvgProj,   f.SheetrockDurationAvgDiv,
    f.TrimDurationAvgProj,        f.TrimDurationAvgDiv,
    f.InteriorDurationAvgProj,    f.InteriorDurationAvgDiv,
    f.FinalDurationAvgProj,       f.FinalDurationAvgDiv,
    f.CloseDurationAvgProj,       f.CloseDurationAvgDiv,

    -- ── Stage tracking ────────────────────────────────────
    f.CurrentStage,
    f.LastCompletedStage,
    f.LastCompletedStageDate

FROM [DataAnalytics].[forecast].[ForecastedStagesOfConstruction] f
WHERE f.CloseDate IS Not NULL 
    AND f.State = 'TX'        
ORDER BY f.JobNumber;



#------------------------------------------------------------------------------------

-- ============================================================
-- FactUnit query for ML training + inference
-- 
-- TWO MODES — change only the WHERE clause at the bottom:
--   Training:  WHERE u.CloseDate IS NOT NULL
--                AND u.FoundationStartDate IS NOT NULL
--   Inference: WHERE u.FoundationStartDate IS NOT NULL
--                AND u.CloseDate IS NULL          -- in-progress jobs
-- ============================================================

-- ============================================================
-- FactUnit query for ML training + inference
-- 
-- TWO MODES — change only the WHERE clause at the bottom:
--   Training:  WHERE u.CloseDate IS NOT NULL
--                AND u.FoundationStartDate IS NOT NULL
--   Inference: WHERE u.FoundationStartDate IS NOT NULL
--                AND u.CloseDate IS NULL          -- in-progress jobs
-- ============================================================

SELECT
    u.JobNumber,
    u.JobReferenceNumber,
    u.State,
    u.County,
    u.City,                             -- NEW v5: used for CityEnc + CityDivisionEnc features
    u.CommunityName,
    u.CommunityId,

    -- Plan details
    u.[Plan],                           -- bracketed: reserved word in SQL
    u.Elevation,
    u.Plansqft,
    u.Lotsqft,
    u.LotType,

    -- Lot characteristics
    u.CornerLotYN,
    u.Swing,
    u.JobSwing,
    u.GarageDrop,
    u.Section,
    u.Block,
    u.Lot,
    u.DivisionName,

    -- Construction metadata
    u.ConstructionCreditDays,
    u.StageOfConstruction,
    u.StageOfConstructionId,

    -- Key milestone dates
    u.PreReleaseDate,
    u.ReleaseDate,
    u.EstimatedReleaseDate,
    u.EstimatedConstructionDate,
    u.ActualConstructionDate,
    u.ActualStartDate,
    u.EstimatedCompletionDate,
    u.ActualCompletionDate,

    -- Actual stage dates from FactUnit (cross-reference)
    -- These are tracked separately from BuildPro stage logic
    u.FoundationStartDate,
    u.FrameDate,
    u.CorniceDate,
    u.MechanicalDate,
    u.SheetrockDate,
    u.TrimDate,
    u.InteriorDate,

    -- Permit delays
    u.CityPermitRequestedDate,
    u.CityPermitReceiveddDate,          -- double d: exact column name
    u.CountyPermitRequesteddDate,       -- double d: exact column name
    u.CountyPermitReceiveddDate,        -- double d: exact column name

    -- Foundation plan delay
    u.FoundationSentDate,
    u.FoundationReceivedDate,

    -- HVAC plan delay (Mechanicals)
    u.HVACManualJDSrequestedDate,
    u.HVACManualJDSReceivedDate,

    -- Selections delay (Trim / Interior)
    u.SelectionsRequestedDate,
    u.SelectionsReceivedDate,

    -- Escrow (Close)
    u.EstimatedEscrowCloseDate,
    u.ActualEscrowCloseDate,

    -- Pricing
    u.BasePrice,
    u.SalesPrice,

    -- Rental flag
    CASE
        WHEN u.ProjectName LIKE '%RENTAL%' THEN 1
        ELSE 0
    END AS IsRental,

    -- Projected close date already in FactUnit
    -- (this is the business-entered estimate, not the 4-month avg model)
    u.ProjectedCloseDate        AS BusinessProjectedCloseDate,
    u.PossibleCloseDate,

    -- ── NEW v5 columns ────────────────────────────────────────────────────────

    -- Geographic / market identifiers
    u.ZipCode,                          -- ZipCodeEnc + ZipRollingAvg features
    u.MarketCode,
    u.MarketName,                       -- MarketEnc feature

    -- Pricing extras
    u.LotPremium,                       -- LotPremium_log feature
    u.NegotiatingAllowance,             -- HasNegotiatingAllowance + log feature
    u.RealtorBonus,                     -- HasRealtorBonus feature

    -- Complexity / custom build flags
    u.AnyAdditionalMasonryRequirement,  -- HasMasonryRequirement feature
    u.AnyCustomRequirement,             -- HasCustomRequirement feature
    u.AnyCOPRequirement,                -- HasCOPRequirement feature

    -- Inventory / spec status
    u.InventoryStatus,                  -- IsSpec feature
    u.UnitStatus,

    -- Plumbing delay
    u.PlumbingRequestedDate,
    u.PlumbingReceivedDate,             -- PlumbingDelayDays feature

    -- Site plan delay
    u.SitePlanRequestedDate,
    u.SitePlanReceivedDate,             -- SitePlanDelayDays feature

    -- Foundation revision delay
    u.FoundationRevisionRequestedDate,
    u.FoundationRevisionReceivedDate,   -- FoundationRevisionDelayDays + HasFoundationRevision

    -- Engineering delay
    u.EngineeringSentDate,
    u.EngineeringReceivedDate,          -- EngineeringDelayDays feature

    -- Developer requirement delay
    u.DeveloperRequestedDate,
    u.DeveloperReceivedDate,            -- DeveloperDelayDays + HasDeveloperRequirement

    -- Thermal plan delay
    u.ThermalPlanRequestedDate,
    u.ThermalPlanReceivedDate,          -- ThermalPlanDelayDays feature

    -- Attic venting plan delay
    u.AtticVentingPlanRequestedDate,
    u.AtticVentingPlanReceivedDate,     -- AtticVentingDelayDays feature

    -- Energy / REZ check delay
    u.EnergyREZCheckRequestedDate,
    u.EnergyREZCheckReceivedDate,       -- EnergyCheckDelayDays feature

    -- Tree permit delay
    u.TreePermitRequestedDate,
    u.TreePermitReceivedDate,           -- TreePermitDelayDays + HasTreePermit

    -- Permit denial flags (HadPermitDenial feature)
    u.CityPermitDenieddDate,            -- double d: exact column name
    u.CountyPermitDeniedDate,

    -- Panel hold (HasPanelHold + PanelHoldDays features)
    u.PanelDate,
    u.PanelOnHoldDate,
    u.PanelOffHoldDate,

    -- No-Rex delay
    u.NoRexRequestedDate,
    u.NoRexReceivedDate                 -- NoRexDelayDays feature

FROM [phazedwserverlessdev].[crossfunction].[FactUnit] u
WHERE u.StreetAddress != '(M)'
      AND u.FoundationStartDate IS NOT NULL
      AND u.CloseDate IS NULL
      AND u.State = 'TX' ;
    -- For training data:  AND u.CloseDate IS NOT NULL
    -- For inference data: AND u.CloseDate IS NULL
