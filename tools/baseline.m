function [X,BL]=baseline(X,bl,dim,method)
%               [X,BL]=baseline(X,bl,dim)
%It baselines X over dimension 'dim'* , using points 'bl'
%according to 'method':
%  -bl= [], data are centered by demeaning 
%  -otherwise
%      method is [] or missing: standard baseline subtraction
%      method is 'zscore': data is normalized using baseline mean and std
%      method is 'percentage': data is percentage change from baseline
%*if 'dim' is missing a default is used [the first matching dimension for
%logical vectors,   the longest dimension for numerical indices].

% Nicola Polizotto (2020), UTHealth, Houston, Texas, USA


if  ~nargin
    help baseline
    return
end

dims=size(X);
if ~exist('dim','var') || isempty(dim)
    if exist('bl','var') && ~isempty(bl) && islogical(bl)
         dim=find(dims==length(bl),1);
    else
         [~,dim]=max(size(X));
    end
end
if ~exist('bl','var') || isempty(bl)
    bl=true(1,dims(dim));
    method=[];
end



dim1=dim-1;
dim2=numel(dims)-dim;

string1=[ repmat(':,',[1 dim1]) ' bl ' repmat(',:',[1 dim2])  ];
if ~exist('method','var') || isempty(method)
    string2=[ 'BL=nanmean(X(' string1 ') ,dim);' ];
    eval(string2)
    X=X-repmat(BL,1+(dims-size(BL)));    
else
    switch upper(method(1))
        case 'Z'
            string2=[ 'BL=nanmean(X(' string1 ') ,dim);' ];
            eval(string2)
            X=X-repmat(BL,1+(dims-size(BL)));              
            string2=[ 'Z=nanstd(X(' string1 ') ,1,dim);' ];
            eval(string2)
            X=X./repmat(Z,1+(dims-size(Z)));
        case 'P'
            string2=[ 'BL=nanmean(X(' string1 ') ,dim);' ];
            eval(string2)
            X=100*(X-repmat(BL,1+(dims-size(BL))))./repmat(BL,1+(dims-size(BL)));                         
        otherwise
            warning('Baselining method could not be recognized, simple baseline subtraction is applied')
    end
end

