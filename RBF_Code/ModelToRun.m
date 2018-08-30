%Model for Input Test Data


%Load test data
load('testdata.csv');

%Load parameters previously saved from Train_Eval_RBF function
load('betas.mat');      %beta values represent 'sigmas' in Gaussian RBF denominator
load('C.mat');          %Centers
load('Weights.mat');    %Weights
load('MeanVec.mat');    %Mean of training data to normalize input
load('StDevVec.mat');   %Standard deviation of training data to normalize input
load('COEFF.mat');      %Coefficients from PCA on training data

%Preprocess incoming test data 
[prepTest] = preprocess(testdata,MeanVec,StDevVec,COEFF);

%Call evaluate function to determine output scores
[scores] = evaluate(Weights, C, betas,prepTest);

%Print scores to command window and write to  outputFile.csv
fprintf('%d\n',scores);
fileID = fopen('outputFile.csv','w');
fprintf(fileID,'%d\n',scores);


%Function to run preprocessed data through RBF filter with weights to get outputs,
%betas (standard distance from each cluster center), and center points.
function [scores] = evaluate(Weights, C, betas,prepTest)

    %Calculate distance of preprocessed input from trained centers
    distance = pdist2(prepTest,C,'mahalanobis');
    
    %Find basis networks for input
    theta = exp((-distance.^2).*betas');
    
    %Calculate output with theta and trained weights
    scores = theta*Weights;
end

%Preprocess data using Training Data's mean, standard deviation, and PCA
%coefficient
function [prepTest] = preprocess(testdata,MeanVec,StDevVec,COEFF)

    %Normalize testdata input
    V = bsxfun(@minus,testdata,MeanVec);
    V2 = bsxfun(@times,V,1./StDevVec);
    
    %Project normalized data onto PCA space
    temp = V2 * COEFF;
    
    %Keep only first principal component - determined from training data
    %trial/error
    prepTest = temp(:,1);
    
end
