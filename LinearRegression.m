classdef LinearRegression < handle & NetworkLayer & Classifier
%TODO: add a function to train directly (closed form)
	properties
		weights;
        bias;
        
        l2_C = 0.01;
        
        dweights;
        dbias;
	end
	
	methods
        function self = LinearRegression()
        end
        
		%those layers with parameters
		function [] = setPar(self, in_size, out_size)		
            self.weights = Utils.randn([in_size, out_size]);
            self.bias = Utils.zeros([out_size,1]);           
            
           self.paramNum = numel(self.weights) + numel(self.bias);
		end
		
        function [] = reset(self)         
            self.weights = Utils.randn(size(self.weights));
            self.bias = Utils.zeros(size(self.bias));
            
            self.dweights = Utils.zeros(size(self.weights));
            self.dbias = Utils.zeros(size(self.bias));
        end
		
		function clearTempData(self)           
            self.IN = [];
            self.OUT = [];
            self.dweights = [];
            self.dbias = [];
        end        		
		
        function param = getParam(self)	       
           param = {self.weights(:); self.bias(:)};
        end
   
        function param = getGradParam(self)	        
            param = {self.dweights(:); self.dbias(:)};
        end
        
        function setParam(self,paramvec)            
            self.weights = reshape(paramvec(1:numel(self.weights)),size(self.weights));
            self.bias = reshape(paramvec(numel(self.weights)+1:end),size(self.bias));
        end
		
        function object = gradCheckObject(self)              
            object = LinearRegression();
        end  
        
		%all layers must implement these
		function [] = fprop(self)
			self.OUT = self.weights'*self.IN + repmat(self.bias,[1 self.numdata]);
		end
        
		function [f derivative] = bprop(self,f,derivative)
            %should be in the top layer
            if isempty(self.dweights) %put this here, only initialize when needed
                self.dweights = Utils.zeros(size(self.weights));
                self.dbias = Utils.zeros(size(self.bias));
            end
            
            if f == 0 %top (this needed to be re-written to be more versatile, like being used in the same time)
                delta = self.OUT - derivative;
                f = 0.5*sum(delta(:).^2);
                derivative = delta;                                                
            end
            
            f = f + 0.5*self.l2_C*norm(self.weights(:))^2;                                   
            
            if ~self.skip_update %update the parameters
                self.dweights = self.IN*derivative' + self.l2_C*self.weights;
                self.dbias = sum(derivative,2);
            end
            if ~self.skip_passdown %compute gradient for lower layers
                derivative = self.weights*derivative;
            end
		end
	end
end