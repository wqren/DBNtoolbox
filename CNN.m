classdef CNN < handle & NetworkLayer
%Convolutional Neural Network based on Mathworks's default filter2
%-------------------
%TODO: larger stride?
%-------------------
%input is width*height*num_channels*num_data
%output is width*height*numunits*num_data
    properties        
        ws;
        numunits;        
        numchannels;
        
        weights;        %ws*ws*numchannels*numunits
        biases;         %numunits*1
        l2_C = 1e-4;        
        
        init_weight = 1e-3;
        
        dweights;
        dbiases;                                 
    end
    
    methods
        function self = CNN(numunits, ws)
            self.ws = ws;
            self.numunits = numunits;
        end
        
        function [] = setPar(self,feadim) %feadim = (W, H, numchannels)		
			self.in_size = feadim; 	            
            self.out_size = [feadim(1)-self.ws+1, feadim(2)-self.ws+1, self.numunits];            
			
            self.numchannels = feadim(3);			
			
            if isempty(self.weights) || nnz(self.in_size ~= feadim)>0
                self.weights = self.init_weight*Utils.randn([self.ws,self.ws,self.numchannels,self.numunits]);
                self.biases = Utils.zeros([self.numunits,1]);
            end
            self.paramNum = numel(self.weights) + numel(self.biases);
		end		
        
        function [] = reset(self)
            self.weights = self.init_weights*Utils.randn(size(self.weights));
            self.biases = Utils.zeros(size(self.biases));
        end
        
        function object = gradCheckObject(self)            
            ws = 3;
            numunits = 2;
            object = CNN(numunits,ws);               
        end         
        
        function fprop(self)            
            self.OUT = Utils.zeros([self.in_size(1)-self.ws+1, self.in_size(2)-self.ws+1, self.numunits, self.numdata]);
            
            %this is very slow!!
            self.OUT = Utils.convInference(self.IN,self.weights);
            
            for u = 1 : self.numunits
                self.OUT(:,:,u,:) = self.OUT(:,:,u,:) + self.biases(u);
            end                        
            
            self.OUT = Utils.sigmoid(self.OUT);
            
        end
        
        function [f derivative] = bprop(self,f,derivative)                            
                if isempty(self.dweights)
                    self.dweights = Utils.zeros(size(self.weights));
                    self.dbiases = Utils.zeros(size(self.biases));
                end
            
                da = self.OUT.*(1-self.OUT).*derivative;         %numunits*numdata                                   
                
                if self.skip_update ~= true
                    self.dweights = Utils.convGradient(self.IN, da);
                    
                    self.dweights = self.dweights + self.l2_C*self.weights;
                    self.dbiases = squeeze(sum(sum(sum(da,1),2),4));
                    f = f + 0.5*self.l2_C*norm(self.weights(:))^2;                              
                end
                               
                if ~self.skip_passdown
                    derivative = Utils.convReconstruct(da, self.weights);                    
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