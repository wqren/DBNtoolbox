classdef NeuralNetwork < handle & NetworkLayer
    properties        
        feadim;
        numunits;
        
        weights;       %feadim*numunits
        biases;         %numunits*1
        l2_C = 1e-4;        
        init_weight = 1e-2;
        
        dweights;
        dbiases;        
    end
    
    methods
        function self = NeuralNetwork(numunits)
            self.numunits = numunits;
            self.out_size = self.numunits;            
        end
        
        function [] = setPar(self,feadim)		
			self.in_size = feadim;
            self.feadim = feadim;
            
			if isempty(self.weights) || nnz(self.in_size ~= feadim)>0                
                self.weights = self.init_weight*Utils.randn([feadim,self.numunits]);
                self.biases = -Utils.ones([self.numunits,1]);
            end
            self.paramNum = numel(self.weights) + numel(self.biases);
		end		
        
        function [] = reset(self)
            self.weights = randn(size(self.weights));
            self.biases = zeros(size(self.biases));
        end
        
        function object = gradCheckObject(self)            
            feadim = 3;
            numunits = 2;
            object = NeuralNetwork(numunits);             
        end         
        
        function fprop(self)            
            self.OUT = Utils.sigmoid(self.weights'*self.IN + repmat(self.biases,[1 self.numdata]));
        end
        
        function [f derivative] = bprop(self,f,derivative)                            
                if isempty(self.dweights)
                    self.dweights = zeros(size(self.weights));
                    self.dbiases = zeros(size(self.biases));
                end
            
                da = self.OUT.*(1-self.OUT).*derivative;         %numunits*numdata                   
                
                if self.skip_update ~= true
                    self.dweights = self.IN*da' + self.l2_C*self.weights;
                    self.dbiases = sum(da,2);
                    f = f + 0.5*self.l2_C*norm(self.weights(:))^2;                              
                end
                
                if ~self.skip_passdown
                    derivative = self.weights*da;
                end                               
        end
        
        function clearTempData(self)
            self.IN = [];
            self.OUT= [];
            self.dweights = [];
            self.dbiases = [];
        end
                        
        
        function param = getParam(self)
            if ~self.skip_update
                param = {self.weights, self.biases};
            else
                param = {};
            end
        end
        
        function param = getGradParam(self)
            if ~self.skip_update
                param = {self.dweights, self.dbiases};              
            else
                param = {};
            end
        end        
        
        function setParam(self,paramvec)
            if ~self.skip_update                
                self.weights = reshape(paramvec(1:numel(self.weights)),size(self.weights));
                self.biases = reshape(paramvec(numel(self.weights)+1:end),size(self.biases));
            end
        end
        
    end    
   
end