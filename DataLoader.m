classdef DataLoader < handle 
%object that load benchmark dataset with static pre-processing functions
    properties
        signature; %used to discriminate dataset and used parameters, also used as save name
        
        Xtrain;
		Xval;
		Xtest;
				
		Ytrain;
		Yval;
		Ytest;		
    end
    
    methods
        function self = DataLoader()
        end
        
        function [] = loadNaturalImage(self)                
            load(fullfile(Config.data_dir_path,'whitened_lc_patches.mat')) % data: 196x160000 natural image patches
            self.Xtrain = data;                        
        end
    end
    
    methods(Static)               
        function [hdr, record] = edfRead(fname, varargin)
        % Read European Data Format file into MATLAB
        %
        % [hdr, record] = edfRead(fname)
        %         Reads data from ALL RECORDS of file fname ('*.edf'). Header
        %         information is returned in structure hdr, and the signals
        %         (waveforms) are returned in structure record, with waveforms
        %         associated with the records returned as fields titled 'data' of
        %         structure record.
        %
        % [...] = edfRead(fname, 'assignToVariables', assignToVariables)
        %         Triggers writing of individual output variables, as defined by
        %         field 'labels', into the caller workspace.
        %
        % FORMAT SPEC: Source: http://www.edfplus.info/specs/edf.html SEE ALSO:
        % http://www.dpmi.tu-graz.ac.at/~schloegl/matlab/eeg/edf_spec.htm
        %
        % The first 256 bytes of the header record specify the version number of
        % this format, local patient and recording identification, time information
        % about the recording, the number of data records and finally the number of
        % signals (ns) in each data record. Then for each signal another 256 bytes
        % follow in the header record, each specifying the type of signal (e.g.
        % EEG, body temperature, etc.), amplitude calibration and the number of
        % samples in each data record (from which the sampling frequency can be
        % derived since the duration of a data record is also known). In this way,
        % the format allows for different gains and sampling frequencies for each
        % signal. The header record contains 256 + (ns * 256) bytes.
        %
        % Following the header record, each of the subsequent data records contains
        % 'duration' seconds of 'ns' signals, with each signal being represented by
        % the specified (in the header) number of samples. In order to reduce data
        % size and adapt to commonly used software for acquisition, processing and
        % graphical display of polygraphic signals, each sample value is
        % represented as a 2-byte integer in 2's complement format. Figure 1 shows
        % the detailed format of each data record.
        %
        % DATA SOURCE: Signals of various types (including the sample signal used
        % below) are available from PHYSIONET: http://www.physionet.org/
        %
        %
        % % EXAMPLE 1:
        % % Read all waveforms/data associated with file 'ecgca998.edf':
        %
        % [header, recorddata] = edfRead('ecgca998.edf');
        %
        % % EXAMPLE 2:
        % % Read records 3 and 5, associated with file 'ecgca998.edf':
        %
        % header = edfRead('ecgca998.edf','AssignToVariables',true);
        % % Header file specifies data labels 'label_1'...'label_n'; these are
        % % created as variables in the caller workspace.
        %
        % Coded 8/27/09 by Brett Shoelson, PhD
        % brett.shoelson@mathworks.com
        % Copyright 2009 - 2012 MathWorks, Inc.

        % HEADER RECORD
        % 8 ascii : version of this data format (0)
        % 80 ascii : local patient identification
        % 80 ascii : local recording identification
        % 8 ascii : startdate of recording (dd.mm.yy)
        % 8 ascii : starttime of recording (hh.mm.ss)
        % 8 ascii : number of bytes in header record
        % 44 ascii : reserved
        % 8 ascii : number of data records (-1 if unknown)
        % 8 ascii : duration of a data record, in seconds
        % 4 ascii : number of signals (ns) in data record
        % ns * 16 ascii : ns * label (e.g. EEG FpzCz or Body temp)
        % ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)
        % ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)
        % ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)
        % ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)
        % ns * 8 ascii : ns * digital minimum (e.g. -2048)
        % ns * 8 ascii : ns * digital maximum (e.g. 2047)
        % ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)
        % ns * 8 ascii : ns * nr of samples in each data record
        % ns * 32 ascii : ns * reserved

        % DATA RECORD
        % nr of samples[1] * integer : first signal in the data record
        % nr of samples[2] * integer : second signal
        % ..
        % ..
        % nr of samples[ns] * integer : last signal

        if nargin > 3
            error('EDFREAD: Too many input arguments.');
        end

        if ~nargin
            error('EDFREAD: Requires at least one input argument (filename to read).');
        end

        if nargin == 1
            assignToVariables = false;
        end

        [fid,msg] = fopen(fname,'r');
        if fid == -1
            error(msg)
        end

        assignToVariables = false; %Default
        for ii = 1:2:numel(varargin)
            switch lower(varargin{ii})
                case 'assigntovariables'
                    assignToVariables = varargin{ii+1};
            end
        end

        % HEADER
        hdr.ver        = str2double(char(fread(fid,8)'));
        hdr.patientID  = fread(fid,80,'*char')';
        hdr.recordID   = fread(fid,80,'*char')';
        hdr.startdate  = fread(fid,8,'*char')';% (dd.mm.yy)
        % hdr.startdate  = datestr(datenum(fread(fid,8,'*char')','dd.mm.yy'), 29); %'yyyy-mm-dd' (ISO 8601)
        hdr.starttime  = fread(fid,8,'*char')';% (hh.mm.ss)
        % hdr.starttime  = datestr(datenum(fread(fid,8,'*char')','hh.mm.ss'), 13); %'HH:MM:SS' (ISO 8601)
        hdr.bytes      = str2double(fread(fid,8,'*char')');
        reserved       = fread(fid,44);
        hdr.records    = str2double(fread(fid,8,'*char')');
        hdr.duration   = str2double(fread(fid,8,'*char')');
        % Number of signals
        hdr.ns    = str2double(fread(fid,4,'*char')');
        for ii = 1:hdr.ns
            hdr.label{ii} = fread(fid,16,'*char')';
        end
        for ii = 1:hdr.ns
            hdr.transducer{ii} = fread(fid,80,'*char')';
        end
        % Physical dimension
        for ii = 1:hdr.ns
            hdr.units{ii} = fread(fid,8,'*char')';
        end
        % Physical minimum
        for ii = 1:hdr.ns
            hdr.physicalMin(ii) = str2double(fread(fid,8,'*char')');
        end
        % Physical maximum
        for ii = 1:hdr.ns
            hdr.physicalMax(ii) = str2double(fread(fid,8,'*char')');
        end
        % Digital minimum
        for ii = 1:hdr.ns
            hdr.digitalMin(ii) = str2double(fread(fid,8,'*char')');
        end
        % Digital maximum
        for ii = 1:hdr.ns
            hdr.digitalMax(ii) = str2double(fread(fid,8,'*char')');
        end
        for ii = 1:hdr.ns
            hdr.prefilter{ii} = fread(fid,80,'*char')';
        end
        for ii = 1:hdr.ns
            hdr.samples(ii) = str2double(fread(fid,8,'*char')');
        end
        for ii = 1:hdr.ns
            reserved    = fread(fid,32,'*char')';
        end
        hdr.label = deblank(hdr.label);
        hdr.units = deblank(hdr.units);


        if nargout > 1 || assignToVariables
            % Scale data (linear scaling)
            scalefac = (hdr.physicalMax - hdr.physicalMin)./(hdr.digitalMax - hdr.digitalMin);
            dc = hdr.physicalMax - scalefac .* hdr.digitalMax;
            
            % RECORD DATA REQUESTED
            tmpdata = struct;
            for recnum = 1:hdr.records
                for ii = 1:hdr.ns
                    % Use a cell array for DATA because number of samples may vary
                    % from sample to sample
                    tmpdata(recnum).data{ii} = fread(fid,hdr.samples(ii),'int16') * scalefac(ii) + dc(ii);
                end
            end
            record = zeros(hdr.ns, hdr.samples(1)*hdr.records);
            
            for ii = 1:numel(hdr.label)
                ctr = 1;
                for jj = 1:hdr.records
                    try
                        record(ii, ctr : ctr + hdr.samples - 1) = tmpdata(jj).data{ii};
                    end
                    ctr = ctr + hdr.samples;
                end
            end
            
            if assignToVariables
                for ii = 1:numel(hdr.label)
                    try
                        eval(['assignin(''caller'',''',hdr.label{ii},''',record(ii,:))'])
                    end
                end
            end
        end
        fclose(fid);       
        end 
    end
end