%%
clear;
clc;
currentdirectory = pwd;

%% Case: FL
num_FL = 63;

fileID = fopen(fullfile(currentdirectory,'\FL\ALL_Cohen_G.txt'),'r');
formatSpec = '%d ';
sizeA = [num_FL Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
FL_All = A';

fileID = fopen(fullfile(currentdirectory,'\FL\Water_Cohen_G.txt'),'r');
formatSpec = '%d ';
sizeA = [num_FL Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
FL_Water = A';
FL_Water_MLR = FL_Water;
FL_Water_rnn = FL_Water;
FL_Water_ann = FL_Water;
FL_Water_raw = FL_Water;

fileID = fopen(fullfile(currentdirectory,'\FL\NoData_Cohen_G.txt'),'r');
formatSpec = '%d ';
sizeA = [num_FL Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
FL_NoData = A';

fileID = fopen(fullfile(currentdirectory,'\FL\PPT_Cohen_G.txt'),'r');
formatSpec = '%f ';
sizeA = [num_FL Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
FL_PPT = A';

fileID = fopen(fullfile(currentdirectory,'\FL\TMT_Cohen_G.txt'),'r');
formatSpec = '%f ';
sizeA = [num_FL Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
FL_TMT = A';

FL_NoDataFrac = FL_NoData./FL_All;
FL_Avail = FL_All(1,:)~=0;
num_FL_new = sum(FL_Avail);

MLR_R2 = zeros(num_FL,1);
RNN_RRMSE = zeros(num_FL,1);
ANN_RRMSE = zeros(num_FL,1);

for iik=1:num_FL
    if FL_Avail(iik)==1
        ii = iik;
        
        %ii=1;
        
        temp_NoData = FL_NoDataFrac(:,ii)>0.05;
        FL_Water_raw(temp_NoData,1) = NaN;
        aa = zeros(360,1);
        for iii = 2:360
            if FL_NoDataFrac(iii,ii)>0.05
                if FL_NoDataFrac(iii-1,ii)<0.05
                    aa(iii) = 1;
                else
                    aa(iii) = aa(iii-1) + 1;
                end
            end
        end
        maxlen = max(aa);
        [r,c] = find(aa==maxlen);
        temp_st = r(1) - maxlen + 1;
        temp_ed = r(1);
        
        % Method 1: Multiple Linear Regression
        
        XX = [FL_Water(temp_st-1:temp_ed-1, ii) FL_PPT(temp_st:temp_ed,ii) FL_TMT(temp_st:temp_ed,ii)];
        YY = FL_Water(temp_st:temp_ed, ii);
        lm = fitlm(XX,YY);
        
        MLR_R2(ii) = lm.Rsquared.Ordinary;
        
        for mi = 2:360
            if FL_NoDataFrac(mi,ii)>0.05
                YYnew = predict(lm,[FL_Water(mi-1,ii) FL_PPT(mi-1,ii) FL_TMT(mi-1,ii)]);
                FL_Water_MLR(mi,ii) = YYnew;
            end
        end
        
        % Method 2: RNN
        
        XTrain = [FL_Water(temp_st:temp_ed-1, ii) FL_PPT(temp_st:temp_ed-1,ii) FL_TMT(temp_st:temp_ed-1,ii)]';
        YTrain = FL_Water(temp_st+1:temp_ed, ii)';
        
        numFeatures = 3;
        numResponses = 1;
        numHiddenUnits = 200;
        
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        
        options = trainingOptions('adam', ...
            'MaxEpochs',250, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',0, ...
            'Plots','training-progress');
        
        net = trainNetwork(XTrain,YTrain,layers,options);
        Yb = predict(net, XTrain);
        RNN_RRMSE(ii) = sqrt(mean((Yb-YTrain).^2))/mean(YTrain);
        
        net = predictAndUpdateState(net,XTrain);
        for mi = 2:360
            if FL_NoDataFrac(mi,ii)>0.05
                [net,YPred] = predictAndUpdateState(net,[FL_Water(mi-1,ii) FL_PPT(mi-1,ii) FL_TMT(mi-1,ii)]');
                FL_Water_rnn(mi,ii) = YPred;
            end
        end
        
        % Method 3: ANN
        
        XXTrain = [FL_Water(temp_st:temp_ed-1, ii) FL_PPT(temp_st:temp_ed-1,ii) FL_TMT(temp_st:temp_ed-1,ii)]';
        YYTrain = FL_Water(temp_st+1:temp_ed, ii)';
        
        net0 = feedforwardnet(10);
        [net0,tr] = train(net0,XXTrain,YYTrain);
        Ya = net0(XXTrain);
        ANN_RRMSE(ii) = sqrt(mean((Ya-YYTrain).^2))/mean(YYTrain);
        
        for mi = 2:360
            if FL_NoDataFrac(mi,ii)>0.05
                YYPred = net0([FL_Water(mi-1,ii) FL_PPT(mi-1,ii) FL_TMT(mi-1,ii)]');
                FL_Water_ann(mi,ii) = YYPred;
            end
        end
    end
end


FL_Water_MLR1 = mean(FL_Water_MLR,2);
FL_Water_rnn1 = mean(FL_Water_rnn,2);
FL_Water_ann1 = mean(FL_Water_ann,2);

startDate = datenum('01-01-1985');
endDate = datenum('01-01-2015');
xData = linspace(startDate,endDate,360);

plot(xData, FL_Water_MLR1, '+r')
hold on
plot(xData, FL_Water_rnn1, 'xb')
hold on
plot(xData, FL_Water_ann1, 'hg')
title('Filled and existed GIW area time-series in FL')
datetick('x','mmmyy','keeplimits')
legend('MLR', 'RNN', 'ANN')
ylabel('GIW pixels for each PRISM grid')

saveas(gcf,'Result_FL.png')
close(gcf)

%% Case: ND

num_ND = 64;

fileID = fopen(fullfile(currentdirectory,'\ND\ALL_Cohen_B.txt'),'r');
formatSpec = '%d ';
sizeA = [num_ND Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
ND_All = A';

fileID = fopen(fullfile(currentdirectory,'\ND\Water_Cohen_B.txt'),'r');
formatSpec = '%d ';
sizeA = [num_ND Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
ND_Water = A';
ND_Water_MLR = ND_Water;
ND_Water_rnn = ND_Water;
ND_Water_ann = ND_Water;
ND_Water_raw = ND_Water;

fileID = fopen(fullfile(currentdirectory,'\ND\NoData_Cohen_B.txt'),'r');
formatSpec = '%d ';
sizeA = [num_ND Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
ND_NoData = A';

fileID = fopen(fullfile(currentdirectory,'\ND\PPT_Cohen_B.txt'),'r');
formatSpec = '%f ';
sizeA = [num_ND Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
ND_PPT = A';

fileID = fopen(fullfile(currentdirectory,'\ND\TMT_Cohen_B.txt'),'r');
formatSpec = '%f ';
sizeA = [num_ND Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
ND_TMT = A';

ND_NoDataFrac = ND_NoData./ND_All;
ND_Avail = ND_All(1,:)~=0;
num_ND_new = sum(ND_Avail);

MLR_R2 = zeros(num_ND,1);
RNN_RRMSE = zeros(num_ND,1);
ANN_RRMSE = zeros(num_ND,1);

for iik=1:num_ND
    if ND_Avail(iik)==1
        ii = iik;
        
        %ii=1;
        
        temp_NoData = ND_NoDataFrac(:,ii)>0.05;
        ND_Water_raw(temp_NoData,1) = NaN;
        aa = zeros(360,1);
        for iii = 2:360
            if ND_NoDataFrac(iii,ii)>0.05
                if ND_NoDataFrac(iii-1,ii)<0.05
                    aa(iii) = 1;
                else
                    aa(iii) = aa(iii-1) + 1;
                end
            end
        end
        maxlen = max(aa);
        [r,c] = find(aa==maxlen);
        temp_st = r(1) - maxlen + 1;
        temp_ed = r(1);
        
        % Method 1: Multiple Linear Regression
        
        XX = [ND_Water(temp_st-1:temp_ed-1, ii) ND_PPT(temp_st:temp_ed,ii) ND_TMT(temp_st:temp_ed,ii)];
        YY = ND_Water(temp_st:temp_ed, ii);
        lm = fitlm(XX,YY);
        
        MLR_R2(ii) = lm.Rsquared.Ordinary;
        
        for mi = 2:360
            if ND_NoDataFrac(mi,ii)>0.05
                YYnew = predict(lm,[ND_Water(mi-1,ii) ND_PPT(mi-1,ii) ND_TMT(mi-1,ii)]);
                ND_Water_MLR(mi,ii) = YYnew;
            end
        end
        
        % Method 2: RNN
        
        XTrain = [ND_Water(temp_st:temp_ed-1, ii) ND_PPT(temp_st:temp_ed-1,ii) ND_TMT(temp_st:temp_ed-1,ii)]';
        YTrain = ND_Water(temp_st+1:temp_ed, ii)';
        
        numFeatures = 3;
        numResponses = 1;
        numHiddenUnits = 200;
        
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        
        options = trainingOptions('adam', ...
            'MaxEpochs',250, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',0, ...
            'Plots','training-progress');
        
        net = trainNetwork(XTrain,YTrain,layers,options);
        Yb = predict(net, XTrain);
        RNN_RRMSE(ii) = sqrt(mean((Yb-YTrain).^2))/mean(YTrain);
        
        net = predictAndUpdateState(net,XTrain);
        for mi = 2:360
            if ND_NoDataFrac(mi,ii)>0.05
                [net,YPred] = predictAndUpdateState(net,[ND_Water(mi-1,ii) ND_PPT(mi-1,ii) ND_TMT(mi-1,ii)]');
                ND_Water_rnn(mi,ii) = YPred;
            end
        end
        
        % Method 3: ANN
        
        XXTrain = [ND_Water(temp_st:temp_ed-1, ii) ND_PPT(temp_st:temp_ed-1,ii) ND_TMT(temp_st:temp_ed-1,ii)]';
        YYTrain = ND_Water(temp_st+1:temp_ed, ii)';
        
        net0 = feedforwardnet(10);
        [net0,tr] = train(net0,XXTrain,YYTrain);
        Ya = net0(XXTrain);
        ANN_RRMSE(ii) = sqrt(mean((Ya-YYTrain).^2))/mean(YYTrain);
        
        for mi = 2:360
            if ND_NoDataFrac(mi,ii)>0.05
                YYPred = net0([ND_Water(mi-1,ii) ND_PPT(mi-1,ii) ND_TMT(mi-1,ii)]');
                ND_Water_ann(mi,ii) = YYPred;
            end
        end
    end
end


ND_Water_MLR1 = mean(ND_Water_MLR,2);
ND_Water_rnn1 = mean(ND_Water_rnn,2);
ND_Water_ann1 = mean(ND_Water_ann,2);

startDate = datenum('01-01-1985');
endDate = datenum('01-01-2015');
xData = linspace(startDate,endDate,360);

plot(xData, ND_Water_MLR1, '+r')
hold on
plot(xData, ND_Water_rnn1, 'xb')
hold on
plot(xData, ND_Water_ann1, 'hg')
title('Filled and existed GIW area time-series in ND')
datetick('x','mmmyy','keeplimits')
legend('MLR', 'RNN', 'ANN')
ylabel('GIW pixels for each PRISM grid')

saveas(gcf,'Result_ND.png')
close(gcf)

