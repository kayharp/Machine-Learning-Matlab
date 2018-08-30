%Train and Evaluate the RBF Network 
%Kristina Harper
%Cand No 178621

%Load data to train model (previously separated into input and output
%files)
load('InputTrainData.csv'); %(10,000x10)
load('OutputTrainData.csv');

%Split train and test data
numVar = length(InputTrainData(:,1));
[trainInd,valInd,testInd] = divideblock(numVar,0.7,0.15,0.15);
trainData = InputTrainData(trainInd,:);
valData = InputTrainData(valInd,:);
testData = InputTrainData(testInd,:);

%Split output targets
targetTrain = OutputTrainData(trainInd,:); 
targetVal = OutputTrainData(valInd,:);
targetTest = OutputTrainData(testInd,:);

%Preprocess Data
[prepData,MeanVec,StDevVec,coef,var,prepTest,prepVal,COEFF,SCORE,LATENT] = Preprocess(trainData,valData,testData);

%Train RBF network
    %Without Preprocessed data
%[Weights,C,basisFunctions,betas] = trainRBFNet(InputTrainData,trainInd,OutputData);
    %With Preprocessed Data
[Weights,C,basisFunctions,betas] = trainPrepRBFNet(prepData,OutputData,trainInd);

%Evaluate RBF network: first function for raw data, second for preprocessed
%data. 
%[scores,accuracy,checkTrainScores,accuracyInput] = evalRBF(InputTrainData,valInd,trainInd,testInd,OutputData,Weights,C,basisFunctions,stdev2);
[scoresVal,scoresTest,accuracy] = evalPrepRBF(prepData,prepTest,prepVal,OutputData,Weights,C,basisFunctions,betas,valInd,testInd,COEFF,SCORE,LATENT,trainInd);

%Print the scores out in command and print to file
fprintf('%d\n',scoresTest);
fileID = fopen('outputFile.txt','w');
fprintf(fileID,'%d\n',scoresTest);
fileID = fopen('outputFile.csv','w');
fprintf(fileID,'%d\n',scoresTest);

%Calculate root mean squared error of validation and test set scores
rmseVal = rootMeanSquaredError(targetVal,scoresVal);
rmseTest = rootMeanSquaredError(targetTest,scoresTest);

%Calculate mean squared error of validation and test set scores
mseVal = immse(targetVal,scoresVal);
mesTest = immse(targetTest,scoresTest);

%Errors
figure()
errors = targetTest-scoresTest;
hist(errors);

% %Plot visualizations of model predictions and actual target output
% figure()
% plot(scores,'r.');hold on;
% plot(OutputTrainData(valInd),'b.');
% title('Mahalanobis Distance');hold on; xlabel('Test Index');ylabel('Output'); 
% legend('Prediction', 'Actual');

%Plot visualization for training data to ensure model can overfit
% figure()
% plot(checkTrainScores,'r.');hold on;
% plot(OutputTrainData(trainInd),'b.');
% title('Mahalanobis Distance');hold on; xlabel('Test Index');ylabel('Output'); 
% legend('Prediction', 'Actual');

%Function to train RBF using preprocessed data
function [Weights,C,basisFunctions,betas] = trainPrepRBFNet(prepData,OutputTrainData,trainInd)

    %First cluster data using KMeans to find cluster structures around centers
    k = 17; %number of centers

    %C is centroid location, IDX is index of data input variable for each structure
    [IDX,C] = kmeans(prepData,k);
    %[IDX,C] = kmeans(prepData,k, 'OnlinePhase', 'on');
    
    %mahalanobis mentioned in lecture as smarter b/c reflects distribution
    %of data...
%    PDistFunc = pdist2(prepData,C,'mahalanobis');
     PDistFunc = pdist2(prepData,C,'mahalanobis');

    %   Calculate beta to use in radial basis function gaussian calculation
    %   (sigma)
    betas = findBetas(prepData,C,IDX);

%   Calculate Gaussian basis functions
    basisFunctions = exp((-PDistFunc.^2).*betas');
    
    trainOutput = OutputTrainData(trainInd);
    
    %Find Weights between hidden and output units to minimize sum-of-squares
    %error between actual and desired outputs. 
    PseudoInv2 = pinv(basisFunctions);
    
    Weights = PseudoInv2*trainOutput;

end

function [scoresVal,scoresTest,accuracy] = evalPrepRBF(prepData,prepTest,prepVal,OutputTrainData,Weights,C,basisFunctions,betas,valInd,testInd,COEFF,SCORE,LATENT,trainInd)
%evaluate model using test data

    centers = C;
%    testTheta = zeros(length(prepVal(:,1)),length(C(:,1)));
    accuracy = zeros(length(valInd),1);
    
    %calculate distance between input data and centers.
%     distance = pdist2(prepVal,centers,'euclidean');
%     checkTrainDistance = pdist2(checkInput, centers,'euclidean');
    distanceVal = pdist2(prepVal,centers,'mahalanobis');
    distanceTest = pdist2(prepTest,centers,'mahalanobis');
    checkTrainDistance = pdist2(prepData, centers,'mahalanobis');
    
    thetaVal = exp((-distanceVal.^2).*betas');
    thetaTest = exp((-distanceTest.^2).*betas');
    checkTrainTheta = exp((-checkTrainDistance.^2).*betas');
    
    %calculate output with theta*weights
    scoresVal = thetaVal*Weights;
    scoresTest = thetaTest*Weights;
    
    %scores = testTheta*Weights;
    OutputTest = OutputTrainData(valInd);
    checkTrainScores = checkTrainTheta*Weights;
    checkTrainOutput = OutputTrainData(trainInd);
   
    
end

function [scores,accuracy,checkTrainScores,accuracyInput] = evalRBF(InputTrainData,valInd,trainInd,testInd,OutputTrainData,Weights,C,basisFunctions,betas)
%evaluate model using test data
    testdata = InputTrainData(valInd,:);

    centers = C;

    accuracy = zeros(length(valInd),1);
    checkInput = InputTrainData(trainInd,:);
    accuracyInput = zeros(length(trainInd),1);
    
    %calculate distance between input data and centers. try euclidean?
    distance = pdist2(testdata,centers,'euclidean');
    checkTrainDistance = pdist2(checkInput, centers,'euclidean');
    
    theta = exp((-distance.^2).*betas');
    checkTrainTheta = exp((-checkTrainDistance.^2).*betas');
    
    
    %calculate output with theta*weights
    scores = theta*Weights;
    %scores = testTheta*Weights;
    OutputTest = OutputTrainData(valInd);
    checkTrainScores = checkTrainTheta*Weights;
    checkTrainOutput = OutputTrainData(trainInd);
    
end

function [Weights,C,basisFunctions,betas] = trainRBFNet(InputTrainData,trainInd,OutputTrainData)

    %First, cluster data using KMeans to find cluster structures around centers
    k = 5; %number of centers
    trainData = InputTrainData(trainInd,:);
    numParams = length(trainData(1,:));

    %C is centroid location, IDX is index of data input variable for each structure
    [IDX,C] = kmeans(trainData,k);
    
    %   Calculate beta to use in radial basis function gaussian calculation
    %   (sigma)
    betas = findBetas(trainData,C,IDX);

%   Calculate distance between each data point and centers

    %mahalanobis mentioned in lecture as smarter b/c reflects distribution
    %of data...
    PDistFunc = pdist2(trainData,C,'euclidean'); 
%     PDistFunc = pdist2(trainData,C,'euclidean');

%   Calculate Gaussian basis functions
    basisFunctions = exp((-PDistFunc.^2).*betas');
    
    trainOutput = OutputTrainData(trainInd);
    %Find Weights between hidden and output units to minimize sum-of-squares
    %error between actual and desired outputs. PInv of phi(basisfunctions) b/c
    %M(numcenters)<N(numinputs)
    PseudoInv2 = pinv(basisFunctions);
    %PseudoInv=pinv(basisFunctions'*basisFunctions)*basisFunctions';
    
    Weights = PseudoInv2*trainOutput;
   % Weights2 = PseudoInv2.*OutputData;
end

%Function to find standard deviation value of members to center
function betas = findBetas(trainData,C,IDX)
    numNeurons = length(C(:,1));
    sigmas = zeros(numNeurons, 1);
    % For each cluster
    for i = 1 : numNeurons
        %Select the next cluster centroid.
        center = C(i, :);
        members = trainData((IDX == i), :);  %Select cluster members
    
        %Subtract the center from each of the member vectors(IDX).
        differences = bsxfun(@minus, members, center);
        
        %Sum of squared differences.
        sqrdDiffs = sum(differences .^ 2, 2);
        
        %Square root to get the Euclidean distance.
        distances = sqrt(sqrdDiffs);

        %Compute the average Euclidean Distance to use as sigma.
        sigmas(i, :) = mean(distances);
    end
    
    %Compute beta values from the sigmas.
    betas = 1 ./ (2 .* sigmas .^ 2);
end


%Function to preprocess data
function [prepData,MeanVec,StDevVec,coef,var1,prepTest,prepVal, COEFF,SCORE,LATENT] = Preprocess(trainData,valData,testData)
    %prepData = zeros(size(trainData));
    feature = zeros(length(trainData(1,:)),1);

    %   Data normalization, then PCA
     MeanVec = mean(trainData);
     StDevVec = std(trainData);
%     T = bsxfun(@minus,trainData,mean(trainData));
%     T2 = bsxfun(@times,trainData,1./std(trainData));
    norm = zscore(trainData);

% Find coefficients of principal components and variances
    [coef var1] = eig(cov(norm)); %first principal component in last COLUMN
    varianceDiag = diag(var1);
    
    %calculate principal components by multiplying standardized data by
    %principal component coefficients
    [COEFF SCORE LATENT] = pca(norm);
    [whitenedData] = whiten(trainData);
    
    %Try different components. 
    %if var(SCORE)<0.6, ignore that feature.
    for it = 1:length(LATENT)
        if LATENT(it) >2 %ignore features with low variance
            feature(it) = 1;
        end
    end
    sum1 = sum(feature);
    prepData = zeros(length(trainData(:,1)),sum1);
    prepTest= zeros(length(testData(:,1)),sum1);
    prepVal= zeros(length(valData(:,1)),sum1);
    
    for its = 1:sum1
        prepData(:,its) = SCORE(:,its);
    end
    
%     %Test to ensure I'm transforming data correctly
%     revTrans = SCORE*COEFF';
%     revTrans1 = bsxfun(@times, revTrans,StDevVec);
%     revTrans2 = bsxfun(@plus,revTrans1,MeanVec);
    
    %normalize test set, project onto new vector space
    T = bsxfun(@minus,testData,MeanVec);
    T2 = bsxfun(@times,T,1./StDevVec);
    tempTest = T2 * COEFF;
    
    V = bsxfun(@minus,valData,MeanVec);
    V2 = bsxfun(@times,V,1./StDevVec);
    tempVal = V2 * COEFF;
    
    for its = 1:sum1
        prepTest(:,its) = tempTest(:,its);
        prepVal(:,its) = tempVal(:,its);
    end
end

function rmse = rootMeanSquaredError(target,prediction)
    rmse = sqrt(mean((target-prediction).^2));
end

%Formulat to whiten normalized data
function [X] = whiten(X)
    epsilon = 0.0001;
    X = bsxfun(@minus, X, mean(X));
    A = X'*X;
    [V,D] = eig(A);
    X = X*V*diag(1./(diag(D)+epsilon).^(1/2))*V';
end