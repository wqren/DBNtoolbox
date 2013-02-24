classdef Softmax < handle & Classifier & NetworkLayer
%softmax classifier, can be used in Deep network or independent (logistic regression) classifier
%WARNING: just changed the initialization, need update to use NN 
    properties        
        weights;       %feadim*numclass
        bias;         %numclass*1
        feadim;
        numclass;
        
        l2_C = 1e-4;                
		l1_C = 0;
        
        dweights;
        dbias;        
    end
    
    methods
        function self = Softmax()                                                                                    
        end
                
        function [] = setPar(self,feadim, numclass)		
			self.in_size = feadim; 	%what are these for?
            self.out_size = numclass;            
			
			self.feadim = feadim;
			self.numclass = numclass;
			self.weights = randn(feadim,numclass);
			self.bias = zeros(numclass,1);
			
            self.paramNum = numel(self.weights) + numel(self.bias);
		end
		
        function [] = reset(self)
            self.weights = randn(size(self.weights));
            self.bias = zeros(size(self.bias));
        end
        
        function checkGradient(self)
            feadim = 3;
            numunits = 2;
            numsamples = 4;
                                   
            sm_gc = Softmax();
			sm_gc.setPar(feadim,numunits);
			sm_gc.l2_C = self.l2_C;
			sm_gc.l1_C = self.l1_C;
            sm_gc.sample_penalty = [1 3 2 1]';
            
			Y = [1;1;2;2];
			y_multi = Utils.num2bin(Y,length(unique(Y)));	              
			X = rand(feadim,numsamples);
			x0 = [sm_gc.weights(:);sm_gc.bias];			
            
            [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) sm_gc.fobj(paramvec, X,y_multi), x0, 1e-5);                       
            fprintf('diff=%g, norm(dbp-dnum)/norm(dbp)=%g\n', d, norm(Utils.vec(dbp-dnum))/(1e-8+norm(Utils.vec(dbp))));
        end
        
        %remove this in the feture------------
        function train(self,X,Y)  
			if isempty(self.weights)
				self.setPar(size(X,1), length(unique(Y)))
			end
			if size(Y,2) == 1
			  y_multi = Utils.num2bin(Y,length(unique(Y)));	              
		    else
		      y_multi = Y; %for multi-labels
			end

			if ~isempty(self.class_penalty)
                self.classPenaltyOnSamples(Y);
            end         
                        						
			  w0 = [self.weights(:) ; self.bias(:)];              
			  w = minFunc(@(paramvec) self.fobj(paramvec, X, y_multi), w0);
			  self.setParam(w);
        end
        %------------------
        
        function [f g] = fobj(self, paramvec, X, y)        
            self.setParam(paramvec);
            self.IN = X;
            self.fprop;
            [f] = self.bprop(0,y);           
            g = [self.dweights(:); self.dbias(:)];
        end               
        
        function [out ] = fprop(self, X)
                if exist('X', 'var') self.IN = X; end                                       
                a = self.weights'*self.IN + repmat(self.bias,1,self.numdata);                
                a = exp(bsxfun(@minus, a, max(a, [], 1))); %buffer                                
                self.OUT = bsxfun(@rdivide, a, sum(a,1));                                   
                if nargout  > 0
                    out = self.OUT;
                end
        end
        
        function [f derivative] = bpropNew(self,f,derivative)
                if isempty(self.dweights)
                    self.dweights = zeros(size(self.weights));
                    self.dbias = zeros(size(self.bias));                    
                end
                
                if f == 0 %top                    
                    %can be more efficient
					% if size(derivative,2) == 1
					  % y = derivative;
                      % y_multi = sparse(y,1:length(y),1,self.numclass,length(y));
                      % y_multi = full(y_multi);                    
					% else 
					  y_multi = derivative;
					% end
					
					fvec = y_multi.*log(self.OUT+1e-8) + (1-y_multi).*log(1-self.OUT+1e-8);
                    derivative = ( -y_multi./(self.OUT+1e-8)  + (1-y_multi)./(1-self.OUT+1e-8) ) / self.numdata;                                            
                    
                    if ~isempty(self.sample_penalty)
						derivative = bsxfun(@times,derivative,self.sample_penalty');  
					    fvec = bsxfun(@times,fvec,self.sample_penalty');                    
					end                    
					f = - mean(sum(fvec));					
                end
                                           
                temp = self.OUT.*derivative;
                da = temp - self.OUT.*(repmat(sum(temp,1),self.numclass,1));
                self.dweights = self.IN * da' + self.l2_C*self.weights + self.l1_C*(2*(self.weights>0)-1);                            
                self.dbias = sum(da,2); 
                
                if ~self.skip_passdown
                    derivative = self.weights*da; % propagate down                                                               
                end

                f = f + 0.5*self.l2_C*norm(self.weights(:))^2 + self.l1_C*sum(abs(self.weights(:))); 
        end
        
        function [f derivative] = bprop(self,f,derivative)
        %only consider positive case
                if isempty(self.dweights)
                    self.dweights = zeros(size(self.weights));
                    self.dbias = zeros(size(self.bias));                    
                end
                
                if f == 0 %top (this need to be re-write to be more versatile)                               
                    %can be more efficient
					% if size(derivative,2) == 1
					  % y = derivative;
                      % y_multi = sparse(y,1:length(y),1,self.numclass,length(y));
                      % y_multi = full(y_multi);                    
					% else 
					  y_multi = derivative;
					% end
                    if ~isempty(self.class_penalty)
                        y_multi = bsxfun(@times,y_multi,self.class_penalty);                    
                    end
                    if ~isempty(self.sample_penalty)
						y_multi = bsxfun(@times,y_multi,self.sample_penalty');                    
					end                    
					f = - mean(sum(y_multi.*log(self.OUT+1e-8)));					
                    derivative = -y_multi.*(1./(self.OUT+1e-8))/self.numdata;                                            
                end
           
                
                
                temp = self.OUT.*derivative;
                da = temp - self.OUT.*(repmat(sum(temp,1),self.numclass,1));
                self.dweights = self.IN * da' + self.l2_C*self.weights + self.l1_C*(2*(self.weights>0)-1);                            
                self.dbias = sum(da,2); 
                
                if ~self.skip_passdown
                    derivative = self.weights*da; % propagate down                                                               
                end

                f = f + 0.5*self.l2_C*norm(self.weights(:))^2 + self.l1_C*sum(abs(self.weights(:))); 
        end
        
        
		function [pred accu] = classify(self, X, y) 
			self.fprop(X);
			accu = [];
            [val, pred] = max(self.OUT,[],1);                            
			pred = pred(:);
			if exist('y','var')
				accu = mean(pred(:)==y(:));            
			end
        end
		     
        
        function clearTempData(self)
            self.IN = [];
            self.OUT= [];
            self.dweights = [];
            self.dbias = [];
        end
        
        
        function param = getParam(self)
	        param = {self.weights, self.bias};
        end
   
        function param = getGradParam(self)
	        param = {self.dweights, self.dbias};
        end
        
        function setParam(self,paramvec)
            self.weights = reshape(paramvec(1:numel(self.weights)),size(self.weights));
            self.bias = reshape(paramvec(numel(self.weights)+1:end),size(self.bias));
        end
        
        function object = gradCheckObject(self)                        
            object = Softmax();             
        end  
    end
    
   
end