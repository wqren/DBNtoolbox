classdef Evaluation < handle
	%util functions related to evaluation
	methods(Static)
	    function [] = runBatch(func, par, job_name)
	        %how to make this thread-safe?
	        %currently only support numeric parameters
	        %par = num_par*1 cell array	        
	        tmp_file = fullfile(Config.tmp_dir_path,job_name); 	        
	        
	        while(true)
                if exist(tmp_file, 'file')
                    %read processed parameters
                    par_runned = importdata(tmp_file);   %num_runned * num_par     
                end
                
                curr_idx_seq = ones(length(par),1);            
                while(true)
                    attempt = zeros(length(par),1);
                    for i = 1 : length(par)
                        attempt(i) = par{i}(curr_idx_seq(i));
                    end
    
                    tried = false;
                    finish = false;                                                            
                    if exist('par_runned','var')
                        for i = 1 : size(par_runned,1)
                            if nnz(attempt ~= par_runned(i,:)') == 0
                                tried = true;
                            end
                        end
                    end
                    
                    if tried == false
                        break;
                    end
                    
                    %otherwise try next 
                    curr_idx_seq(1) = curr_idx_seq(1) + 1;
                    if length(par) == 1 && curr_idx_seq(1) > length(par{1})
                        finish = true;
                    end
                    for i = 1 : length(par)-1
                        if curr_idx_seq(i) > length(par{i})
                            curr_idx_seq(i) = 1;
                            curr_idx_seq(i+1) = curr_idx_seq(i+1) + 1;
                            if curr_idx_seq(end) > length(par{end})
                                finish = true;                            
                            end 
                        end 
                    end
                    
                    if finish 
                        break;
                    end                    
                end
                
                if finish
                    break;
                end
                
                %write attempt to file
                fid = fopen(tmp_file, 'a');
                for i = 1 : length(attempt)
                    fprintf(fid,'%f ', attempt(i));
                end
                fprintf(fid,'\n');
                fclose(fid);
                
                %run
                fprintf('start running...\n');
                attempt
                func
                output = func(attempt);
                fid = fopen(fullfile(Config.result_dir_path,job_name), 'a');
                for i = 1 : length(attempt)
                    fprintf(fid,'%f ', attempt(i));
                end
                fprintf(fid,', o/p: %s\n', output);
                fclose(fid);
            end
	    end
	    
	    %------------validation----------------
		function [classifier result] = validate(classifiers, data, val_idx, metric)
			%data need to contain Xtrain, Ytrain, Xval, Yval (if isempty(val_idx) ) 
			%apply validation to choose from classifiers						
			if ~exist('metric', 'var') || isempty(metric)
				metric = @Evaluation.accuracy;
			end
			
			train_idx = [1:length(data.Ytrain)];
			train_idx(val_idx) = [];
			
			for i = 1 : length(classifiers)
				classifiers{i}.train(data.Xtrain(:,train_idx), data.Ytrain(train_idx));
			end
			
			result = zeros(length(classifiers),1);					
			for i = 1 : length(classifiers)
				if ~isempty(val_idx)
					pred = classifiers{i}.classify(data.Xtrain(:,val_idx));
					result(i) = metric(pred, data.Ytrain(val_idx));
				else
					pred = classifiers{i}.classify(data.Xval);
					result(i) = metric(pred, data.Yval);
				end
			end
			
			[~,bestidx] = max(result);			
			classifier = classifiers{bestidx};
		end
		
		function [train_idx val_idx] = getValIdx(numsamples, K)
			train_idx = 1:numsamples;
			val_idx = randperm(numsamples);
			val_idx = val_idx(1:round(numsamples/K));
			train_idx(val_idx) = [];
		end
		
		function [Data] = separateValData(oriXtrain, oriYtrain, validx)
		    Data.Xval = oriXtrain(:,validx);
		    Data.Yval = oriYtrain(validx);
		    Data.Xtrain = oriXtrain;
		    Data.Xtrain(:,validx) = [];
		    Data.Ytrain = oriYtrain;
		    Data.Ytrain(validx) = [];
		end
		
		%==================evaluation metrics=====================
		function [accu] = accuracy(pred, Ytest)
			accu = mean(pred == Ytest);
		end
		
		function [f] = macroF1(pred, Ytest)
			[recall, prec] = Evaluation.precRecall(pred,Ytest);
			recall = mean(recall);
			prec = mean(prec);
			f = 2*recall*prec / (prec + recall);
		end
		
		function [sensitivity, selectivity, specificity] = precRecall(pred, Ytest)			
			%recall, precision, recall_negative
			unique_idx = unique(Ytest);		
			sensitivity = zeros(length(unique_idx),1);
			selectivity = zeros(length(unique_idx),1);
			specificity = zeros(length(unique_idx),1);
						
			for i = 1 : length(unique_idx)									
				tp = sum((pred == unique_idx(i)) & (Ytest == unique_idx(i)));
				tn = sum((pred ~= unique_idx(i)) & (Ytest ~= unique_idx(i)));
				fp = sum((pred == unique_idx(i)) & (Ytest ~= unique_idx(i)));
				fn = sum((pred ~= unique_idx(i)) & (Ytest == unique_idx(i)));
				
				sensitivity(i) = tp/(tp+fn);
				selectivity(i) = tp/(tp+fp);
				specificity(i) = tn/(tn+fp);				
			end			
		end
		
		function [] = precRecallCurve(posterior, category)
			if min(unique(category)) ~= 0 || max(unique(category)) ~= 1
				error('category must be binary');
			end
			[~,ind] = sort(posterior,'descend'); 
			roc_y = category(ind);
						
			prec = Utils.vec(cumsum(roc_y == 1))./[1:length(roc_y)]';
			recall = cumsum(roc_y == 1) / sum(roc_y == 1);
			
			plot(recall, prec);
			xlabel('recall');
			ylabel('precision');
			title(['precision recall curve']);
		end
        
        function [AUC] = AUPRC(score, label)
            %label must be binary
            [X,Y,T,AUC] = perfcurve(label,score,true,'xCrit','TPR','yCrit','PPV');
        end

%test this in the future		
%		function [auc] = AUROC(posterior, category, plot_result) %category should be binary
%			if min(unique(category)) ~= 0 || max(unique(category)) ~= 1
%				error('category must be binary');
%			end
%			if ~exist('plot_result','var')
%				plot_result = false;
%			end
%			[~,ind] = sort(posterior,'descend'); 
%			roc_y = category(ind);
%			stack_x = cumsum(roc_y == 0)/sum(roc_y == 0);
%			stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
%			auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
%			
%			if (plot_result)
%				plot(stack_x, stack_y);
%				xlabel('False Positive Rate');
%				ylabel('True Positive Rate');
%				title(['ROC curve of (AUC = ' num2str(auc) ' )']);
%			end
%		end				
				
	end
end