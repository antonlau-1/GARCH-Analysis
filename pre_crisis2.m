% Clean the returns data
returns = SP500PreCrisisHistorical.LogDifference;
returns = returns(~isnan(returns));
returns = returns(:);

% Get the dates
% Clean the returns data
returns = SP500PreCrisisHistorical.LogDifference;
returns = returns(~isnan(returns));
returns = returns(:);

% Get the dates
dates = SP500PreCrisisHistorical.Date;
dates = dates(~isnan(SP500PreCrisisHistorical.LogDifference)); % Ensure dates match the non-NaN returns

% Estimate GARCH(1,1) model with Gaussian distribution
garchModel = garch(1,1);
[estModel, estParamCov, logL] = estimate(garchModel, returns);
disp('GARCH(1,1) Parameters (Gaussian):');
disp(estModel);

% Estimate GARCH(1,1) model with Student's t-distribution
garchModel_t = garch(1,1);
garchModel_t.Distribution = 't';
[estModel_t, estParamCov_t, logL_t] = estimate(garchModel_t, returns);
disp('GARCH(1,1) Parameters (Student''s t):');
disp(estModel_t);
disp(['Degrees of Freedom: ', num2str(estModel_t.Distribution.DoF)]);

% Infer the conditional variance using both estimated GARCH models
condVarGARCH = infer(estModel, returns);
condVarGARCH_t = infer(estModel_t, returns);

% Ensure the length of conditional variance matches the length of returns
if length(condVarGARCH) ~= length(returns)
    condVarGARCH = condVarGARCH(end-length(returns)+1:end);
    condVarGARCH_t = condVarGARCH_t(end-length(returns)+1:end);
    plotDates = dates(2:end);
else
    plotDates = dates;
end

% Plot the results for GARCH(1,1)
figure;
subplot(3,1,1);
plot(dates, returns);
title('S&P 500 Log Returns - Pre-crisis');
datetick('x', 'dd-mmm-yyyy'); % Format date labels
subplot(3,1,2);
plot(plotDates, sqrt(condVarGARCH));
title('GARCH(1,1) Conditional Volatility (Gaussian)');
datetick('x', 'dd-mmm-yyyy'); % Format date labels
subplot(3,1,3);
plot(plotDates, sqrt(condVarGARCH_t));
title('GARCH(1,1) Conditional Volatility (Student''s t)');
datetick('x', 'dd-mmm-yyyy'); % Format date labels

% Estimate EGARCH(1,1) model with Gaussian distribution
egarchModel = egarch(1,1);
[estModelE, estParamCovE, logLE] = estimate(egarchModel, returns);
disp('EGARCH(1,1) Parameters (Gaussian):');
disp(estModelE);

% Estimate EGARCH(1,1) model with Student's t-distribution
egarchModel_t = egarch(1,1);
egarchModel_t.Distribution = 't';
[estModelE_t, estParamCovE_t, logLE_t] = estimate(egarchModel_t, returns);
disp('EGARCH(1,1) Parameters (Student''s t):');
disp(estModelE_t);
disp(['Degrees of Freedom: ', num2str(estModelE_t.Distribution.DoF)]);

% Infer the conditional variance using both estimated EGARCH models
condVarEGARCH = infer(estModelE, returns);
condVarEGARCH_t = infer(estModelE_t, returns);

% Ensure the length of conditional variance matches the length of returns
if length(condVarEGARCH) ~= length(returns)
    condVarEGARCH = condVarEGARCH(end-length(returns)+1:end);
    condVarEGARCH_t = condVarEGARCH_t(end-length(returns)+1:end);
    plotDatesE = dates(2:end);
else
    plotDatesE = dates;
end

% Plot the results for EGARCH(1,1)
figure;
subplot(3,1,1);
plot(dates, returns);
title('S&P 500 Log Returns - Pre-crisis');
datetick('x', 'dd-mmm-yyyy'); % Format date labels
subplot(3,1,2);
plot(plotDatesE, sqrt(condVarEGARCH));
title('EGARCH(1,1) Conditional Volatility (Gaussian)');
datetick('x', 'dd-mmm-yyyy'); % Format date labels
subplot(3,1,3);
plot(plotDatesE, sqrt(condVarEGARCH_t));
title('EGARCH(1,1) Conditional Volatility (Student''s t)');
datetick('x', 'dd-mmm-yyyy'); % Format date labels

% Calculate standardized residuals for GARCH (Gaussian)
standardizedResiduals = (returns - condVarGARCH.^0.5) ./ condVarGARCH.^0.5;

% Perform Ljung-Box test for GARCH (Gaussian)
[h, pValue, stat, cValue] = lbqtest(standardizedResiduals);
disp('Ljung-Box Test for GARCH(1,1) standardized residuals (Gaussian):');
disp(['h (reject null hypothesis): ', num2str(h)]);
disp(['p-value: ', num2str(pValue)]);
disp(['Test statistic: ', num2str(stat)]);
disp(['Critical value: ', num2str(cValue)]);

% Calculate standardized residuals for GARCH (Student's t)
standardizedResiduals_t = (returns - condVarGARCH_t.^0.5) ./ condVarGARCH_t.^0.5;

% Perform Ljung-Box test for GARCH (Student's t)
[h_t, pValue_t, stat_t, cValue_t] = lbqtest(standardizedResiduals_t);
disp('Ljung-Box Test for GARCH(1,1) standardized residuals (Student''s t):');
disp(['h (reject null hypothesis): ', num2str(h_t)]);
disp(['p-value: ', num2str(pValue_t)]);
disp(['Test statistic: ', num2str(stat_t)]);
disp(['Critical value: ', num2str(cValue_t)]);

% Calculate standardized residuals for EGARCH (Gaussian)
standardizedResidualsE = (returns - condVarEGARCH.^0.5) ./ condVarEGARCH.^0.5;

% Perform Ljung-Box test for EGARCH (Gaussian)
[hE, pValueE, statE, cValueE] = lbqtest(standardizedResidualsE);
disp('Ljung-Box Test for EGARCH(1,1) standardized residuals (Gaussian):');
disp(['h (reject null hypothesis): ', num2str(hE)]);
disp(['p-value: ', num2str(pValueE)]);
disp(['Test statistic: ', num2str(statE)]);
disp(['Critical value: ', num2str(cValueE)]);

% Calculate standardized residuals for EGARCH (Student's t)
standardizedResidualsE_t = (returns - condVarEGARCH_t.^0.5) ./ condVarEGARCH_t.^0.5;

% Perform Ljung-Box test for EGARCH (Student's t)
[hE_t, pValueE_t, statE_t, cValueE_t] = lbqtest(standardizedResidualsE_t);
disp('Ljung-Box Test for EGARCH(1,1) standardized residuals (Student''s t):');
disp(['h (reject null hypothesis): ', num2str(hE_t)]);
disp(['p-value: ', num2str(pValueE_t)]);
disp(['Test statistic: ', num2str(statE_t)]);
disp(['Critical value: ', num2str(cValueE_t)]);

% Compare log-likelihoods
disp('Log-likelihood comparison:');
disp(['GARCH(1,1) Gaussian: ', num2str(logL)]);
disp(['GARCH(1,1) Student''s t: ', num2str(logL_t)]);
disp(['EGARCH(1,1) Gaussian: ', num2str(logLE)]);
disp(['EGARCH(1,1) Student''s t: ', num2str(logLE_t)]);
dates = SP500PreCrisisHistorical.Date;
dates = dates(~isnan(SP500PreCrisisHistorical.LogDifference)); % Ensure dates match the non-NaN returns

% Estimate GARCH(1,1) model
garchModel = garch(1,1);
[estModel, estParamCov, logL] = estimate(garchModel, returns);
disp('GARCH(1,1) Parameters:');
disp(estModel);

% Infer the conditional variance using the estimated GARCH model
condVarGARCH = infer(estModel, returns);

% Ensure the length of conditional variance matches the length of returns
if length(condVarGARCH) ~= length(returns)
    condVarGARCH = condVarGARCH(end-length(returns)+1:end);
    plotDates = dates(2:end);
else
    plotDates = dates;
end

% Plot the results for GARCH(1,1)
figure;
subplot(2,1,1);
plot(dates, returns);
title('S&P 500 Log Returns - Pre-crisis');
datetick('x', 'dd-mmm-yyyy'); % Format date labels
subplot(2,1,2);
plot(plotDates, sqrt(condVarGARCH));
title('GARCH(1,1) Conditional Volatility');
datetick('x', 'dd-mmm-yyyy'); % Format date labels

% Estimate EGARCH(1,1) model
egarchModel = egarch(1,1);
[estModelE, estParamCovE, logLE] = estimate(egarchModel, returns);
disp('EGARCH(1,1) Parameters:');
disp(estModelE);

% Infer the conditional variance using the estimated EGARCH model
condVarEGARCH = infer(estModelE, returns);

% Ensure the length of conditional variance matches the length of returns
if length(condVarEGARCH) ~= length(returns)
    condVarEGARCH = condVarEGARCH(end-length(returns)+1:end);
    plotDatesE = dates(2:end);
else
    plotDatesE = dates;
end

% Plot the results for EGARCH(1,1)
figure;
subplot(2,1,1);
plot(dates, returns);
title('S&P 500 Log Returns - Pre-crisis');
datetick('x', 'dd-mmm-yyyy'); % Format date labels
subplot(2,1,2);
plot(plotDatesE, sqrt(condVarEGARCH));
title('EGARCH(1,1) Conditional Volatility');
datetick('x', 'dd-mmm-yyyy'); % Format date labels

% Calculate standardized residuals for GARCH(1,1)
standardizedResiduals = (returns - condVarGARCH.^0.5) ./ condVarGARCH.^0.5;

% % Perform Ljung-Box test for GARCH(1,1)
% [h, pValue, stat, cValue] = lbqtest(standardizedResiduals);
% disp('Ljung-Box Test for GARCH(1,1) standardized residuals:');
% disp(['h (reject null hypothesis): ', num2str(h)]);
% disp(['p-value: ', num2str(pValue)]);
% disp(['Test statistic: ', num2str(stat)]);
% disp(['Critical value: ', num2str(cValue)]);
% 
% % Calculate standardized residuals for EGARCH(1,1)
% standardizedResidualsE = (returns - condVarEGARCH.^0.5) ./ condVarEGARCH.^0.5;
% 
% % Perform Ljung-Box test for EGARCH(1,1)
% [hE, pValueE, statE, cValueE] = lbqtest(standardizedResidualsE);
% disp('Ljung-Box Test for EGARCH(1,1) standardized residuals:');
% disp(['h (reject null hypothesis): ', num2str(hE)]);
% disp(['p-value: ', num2str(pValueE)]);
% disp(['Test statistic: ', num2str(statE)]);
% disp(['Critical value: ', num2str(cValueE)]);
% 
% % QQ plots
% figure;
% qqplot(standardizedResiduals);
% title('QQ Plot of Standardized Residuals - GARCH(1,1)');
% 
% figure;
% qqplot(standardizedResidualsE);
% title('QQ Plot of Standardized Residuals - EGARCH(1,1)');
