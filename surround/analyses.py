def compare_to_experiment(frequencies, spectra, space_h=None, proj_h=None, space_a=None, proj_a=None, 
        inputNoise=0.1, outputNoise=0.4, center_weighting=2.1, surround_weighting=0.1,
        horz_weighting=0.5, ama_weighting=0.5, center_width=.5, interpolation='fit', 
        numPoints=1000, returnFlag=False, plotFlag='aggregate', verbose=True, xlimit=None, aggregateColor='c', statistics=False):
    ''' Compare ideal infomax filter to experimental projective fields.
    INPUTS:
    frequencies: np array of spatial frequencies corresponding to spectra
    spectra: np array of original power spectra
    space_h: space in degrees corresponding to horizontal cell projective field
    proj_h: horizontal cell projective field
    space_a: space in degrees corresponding to amacrine cell projective field
    proj_a: amacrine cell projective field
    
    RETURNS:
    frequencies_ideal, ideal_filter, frequencies_expt, expt_filter
    '''

    ###### EXPERIMENTAL ######
    # make surround
    if space_h is None:
        horz_pfs = get_horizontal_projective_field()
        ama_pfs  = get_amacrine_projective_field()

        # get means of the projective fields
        space_h, horz_pf, horz_sem = get_mean(horz_pfs)
        space_a, ama_pf, ama_sem   = get_mean(ama_pfs)

        # interpolate horz and ama to get a unified space; mode='valid'
        horz_interp = interp1d(space_h, horz_pf, kind='slinear')
        ama_interp  = interp1d(space_a, ama_pf, kind='slinear')
        space       = np.linspace(np.max([np.min(space_h), np.min(space_a)]), np.min([np.max(space_h), np.max(space_a)]), 100)

        # project interpolations on unified space
        horz_pf     = horz_interp(space)
        ama_pf      = ama_interp(space)

        ## FOR DEBUGGING ##
        #return space, horz_pf, ama_pf

        # set sampling rate
        spacing = space[-1] - space[-2]
    
    # make surround
    surround       = horz_weighting * horz_pf + ama_weighting * ama_pf

    # make center
    if center_width is None:
        center = center_weighting * np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
    else:
        center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

    # put them together
    if len(center.shape) > 1:
        center = center.squeeze()
    rf = center_weighting * center + surround_weighting * surround

    # Amplitude Spectrum of RF
    rf_f_two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
    n = len(rf_f_two_sided)
    if n % 2 == 0:
        rf_f_one_sided = rf_f_two_sided[:(int(n/2) + 1)]
    else:
        rf_f_one_sided = rf_f_two_sided[:(int((n-1)/2) + 1)]
    rf_freqs_one_sided = np.linspace(0, 1./(2*spacing), len(rf_f_one_sided))

    
    ###### IDEAL ######
    if interpolation == 'linear':
        naturalInterpolation = interp1d(frequencies, spectra, kind='linear')
        moreFreqs   = np.linspace(0., frequencies[-1], numPoints)
        moreSpectra = naturalInterpolation(moreFreqs)
        idealFilter = unique_soln(moreSpectra, inputNoise, outputNoise, verbose=verbose) # spectra, input_noise, output_noise
    elif interpolation == 'cubic':
        naturalInterpolation = interp1d(frequencies, spectra, kind='cubic')
        moreFreqs   = np.linspace(0., frequencies[-1], numPoints)
        moreSpectra = naturalInterpolation(moreFreqs)
        idealFilter = unique_soln(moreSpectra, inputNoise, outputNoise, verbose=verbose) # spectra, input_noise, output_noise
    elif interpolation == 'fit':
        # power law
        def func(x, a, b):
            return a/(x**b) # adding a constant tends to make moreSpectra have negative numbers

        popt, pcov  = curve_fit(func, frequencies[1:], spectra[1:])
        moreFreqs   = np.linspace(0., frequencies[-1], numPoints)[1:]
        moreSpectra = func(moreFreqs, *popt)
        idealFilter = unique_soln(moreSpectra, inputNoise, outputNoise, verbose=verbose) # spectra, input_noise, output_noise
    elif interpolation is None:
        idealFilter = unique_soln(spectra, inputNoise, outputNoise, verbose=verbose)
        
    if plotFlag == 'aggregate':
        # first we need to compute the fft for all combinations of horz and amacrine cell
        horz_pfs = get_horizontal_projective_field()
        horz_pfs = get_interp(horz_pfs, mode='valid')
        ama_pfs  = get_amacrine_projective_field()
        ama_pfs  = get_interp(ama_pfs, mode='valid')

        min_space = np.max([np.max([np.min(x) for x,y in horz_pfs]), np.max([np.min(x) for x,y in ama_pfs])])
        max_space = np.min([np.min([np.max(x) for x,y in horz_pfs]), np.min([np.max(x) for x,y in ama_pfs])])
        space     = np.linspace(min_space, max_space, 100)
        rf_ffts   = []

        for fh, hp in horz_pfs:
            for ah, ap in ama_pfs:
                horz_interp = interp1d(fh, hp, kind='slinear')
                ama_interp  = interp1d(ah, ap, kind='slinear')

                surround = horz_weighting * horz_interp(space) + ama_weighting * ama_interp(space)

                # make center
                if center_width is None:
                    center = center_weighting * np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
                else:
                    center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

                # put them together
                if len(center.shape) > 1:
                    center = center.squeeze()
                rf = center_weighting * center + surround_weighting * surround

                # Amplitude Spectrum of RF
                two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
                n = len(two_sided)
                if n % 2 == 0:
                    rf_ffts.append(two_sided[:(int(n/2) + 1)])
                else:
                    rf_ffts.append(two_sided[:(int((n-1)/2) + 1)])

        rf_fft_f    = np.linspace(0, 1./(2.*(space[-1]-space[-2])), len(rf_ffts[0]))
        rf_ffts_err = sem(rf_ffts)
        scaling = np.nanmax(rf_f_one_sided)
        scaling_err = np.nanmax(np.mean(rf_ffts, axis=0))

        #for rf_fft in rf_ffts:
        #    freqs_one_sided = np.linspace(0, 1./(2*(space[-1]-space[-2])), len(rf_fft))
        #    plt.plot(freqs_one_sided, rf_fft/scaling_err, 'c', linewidth=2)

        plt.plot(rf_freqs_one_sided, (rf_f_one_sided/scaling), aggregateColor, linewidth=9, alpha=0.8)
        plt.errorbar(rf_freqs_one_sided, rf_f_one_sided/scaling, yerr=rf_ffts_err/scaling_err, color=aggregateColor, alpha=0.7, linewidth=9, capthick=5, capsize=5)
        plt.plot(moreFreqs, (idealFilter/np.nanmax(idealFilter)), color='#6699ff', alpha=0.7, linewidth=9)
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left', 'bottom'])
        plt.ylim([0, 1.2])
        if xlimit:
            plt.xlim(xlimit)
    elif plotFlag == 'separate':

        freq_h, horz_fft = get_fft([(space, horz_pf)], mode='amplitude')[0]
        freq_a, ama_fft  = get_fft([(space, ama_pf)], mode='amplitude')[0]

        flag = 'exp'

        if flag == 'interp':
            # interpolate horizontal and amacrine projective field filters in frequency space
            horz_interpolate = interp1d(freq_h, 1-horz_fft/np.max(horz_fft), kind='slinear')
            ama_interpolate  = interp1d(freq_a, 1-ama_fft/np.max(ama_fft), kind='slinear')

            horz_interpolate_x = np.linspace(freq_h[0], freq_h[-1], numPoints)
            ama_interpolate_x  = np.linspace(freq_a[0], freq_a[-1], numPoints)

            horz_fit = horz_interpolate(horz_interpolate_x)
            ama_fit  = ama_interpolate(ama_interpolate_x)

        elif flag == 'exp':
            def func(x, a, b, c):
                # exponential
                y = a * np.exp(-b * x) + c

                # power law
                #y = a / x**b + c
                #y[x==0] = np.max(y[np.isfinite(y)])

                return 1 - y/np.max(y)

            popt_horz, pcov_horz = curve_fit(func, freq_h, 1-horz_fft/np.max(horz_fft))
            popt_ama, pcov_ama   = curve_fit(func, freq_a, 1-ama_fft/np.max(ama_fft))
            
            horz_interpolate_x = np.linspace(freq_h[0], freq_h[-1], numPoints)
            ama_interpolate_x  = np.linspace(freq_a[0], freq_a[-1], numPoints)
            
            horz_fit = func(horz_interpolate_x, *popt_horz)
            ama_fit  = func(ama_interpolate_x, *popt_ama)

                
        plt.plot(moreFreqs, idealFilter/np.nanmax(idealFilter), 'b-', linewidth=3, alpha=0.5) 
        plt.plot(horz_interpolate_x, horz_fit, 'g', linewidth=2, alpha=0.8)
        plt.plot(ama_interpolate_x, ama_fit, 'c', linewidth=2, alpha=0.8)
        plt.plot(freq_h, 1 - horz_fft/np.max(horz_fft), 'g.', alpha=0.8, markersize=10)
        plt.plot(freq_a, 1 - ama_fft/np.max(ama_fft), 'c.', alpha=0.8, markersize=10)
        if xlimit:
            plt.xlim(xlimit)

    elif plotFlag == 'many':
        #input_noises  = [0.1, 0.2, 0.3] #[0.05, 0.1, 0.15, 0.3, 0.4, 0.6]
        #output_noises = [0.05, 0.1, 0.2, 0.3] #[0.05, 0.1, 0.15, 0.3, 0.4, 0.6]
        noises = [(0.05, 0.1), (0.12, 0.35), (0.08, 0.2), (0.2, 0.42)]
        viewing_cutoff   = 0.25
        
        #for inn in input_noises:
        #    for outn in output_noises:
        for inn, outn in noises:
            if inn < 0.06:
                relevance_cutoff = 0.3
            elif inn < .1:
                relevance_cutoff = 0.22
            elif inn < .2:
                relevance_cutoff = 0.15
            else:
                relevance_cutoff = 0.12
            freq_ideal, filt_ideal, _, _ = compare_to_experiment(frequencies, spectra, inputNoise=inn, outputNoise=outn, verbose=False, returnFlag=True, numPoints=5000, plotFlag=False)
            fitted_rf, popt = fit_ideal(freq_ideal[freq_ideal < relevance_cutoff], filt_ideal[freq_ideal < relevance_cutoff]/np.nanmax(filt_ideal[freq_ideal < relevance_cutoff]), returnFlag='interp')
            freqs = freq_ideal[freq_ideal < viewing_cutoff]
            if outn == 0.35:
                plt.plot(freqs, fitted_rf(freqs, *popt), 'c', linewidth=5, alpha=0.5)
            else:
                plt.plot(freqs, fitted_rf(freqs, *popt), 'k', linewidth=5, alpha=0.3)
            plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color='#6699ff', linewidth=11, alpha=0.5)

            # get errorbars
            # first we need to compute the fft for all combinations of horz and amacrine cell
            horz_pfs = get_horizontal_projective_field()
            horz_pfs = get_interp(horz_pfs, mode='valid')
            ama_pfs  = get_amacrine_projective_field()
            ama_pfs  = get_interp(ama_pfs, mode='valid')

            min_space = np.max([np.max([np.min(x) for x,y in horz_pfs]), np.max([np.min(x) for x,y in ama_pfs])])
            max_space = np.min([np.min([np.max(x) for x,y in horz_pfs]), np.min([np.max(x) for x,y in ama_pfs])])
            space     = np.linspace(min_space, max_space, 100)
            rf_ffts   = []

            horz_weighting, ama_weighting, center_weighting, surround_weighting, center_width = popt

            for fh, hp in horz_pfs:
                for ah, ap in ama_pfs:
                    horz_interp = interp1d(fh, hp, kind='slinear')
                    ama_interp  = interp1d(ah, ap, kind='slinear')

                    surround = horz_weighting * horz_interp(space) + ama_weighting * ama_interp(space)

                    # make center
                    if center_width is None:
                        center = center_weighting * np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
                    else:
                        center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

                    # put them together
                    if len(center.shape) > 1:
                        center = center.squeeze()
                    rf = center_weighting * center + surround_weighting * surround

                    # Amplitude Spectrum of RF
                    two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
                    n = len(two_sided)
                    if n % 2 == 0:
                        rf_ffts.append(two_sided[:(int(n/2) + 1)])
                    else:
                        rf_ffts.append(two_sided[:(int((n-1)/2) + 1)])

            rf_ffts_err = sem(rf_ffts)
            scaling = np.nanmax(rf_f_one_sided)
            scaling_err = np.nanmax(np.mean(rf_ffts, axis=0))
            err_interp = interp1d(np.linspace(0, 1./(2.*(space[-1]-space[-2])), len(rf_ffts_err)), rf_ffts_err/scaling_err)

            #for rf_fft in rf_ffts:
            #    freqs_one_sided = np.linspace(0, 1./(2*(space[-1]-space[-2])), len(rf_fft))
            #    plt.plot(freqs_one_sided, rf_fft/scaling_err, 'c', linewidth=2)

            if outn == 0.35:
                plt.errorbar(freqs, fitted_rf(freqs, *popt), yerr=err_interp(freqs), color='c', alpha=0.5, linewidth=5, capthick=4, capsize=2)
            else:
                plt.errorbar(freqs, fitted_rf(freqs, *popt), yerr=err_interp(freqs), color='k', alpha=0.5, linewidth=5, capthick=4, capsize=2)


            
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left', 'bottom'])
        plt.ylim([0, 1.2])
        if xlimit:
            plt.xlim(xlimit)

    elif plotFlag == 'many-one':


        noises = [(0.05, 0.1), (0.12, 0.35), (0.08, 0.2), (0.2, 0.42)]
        viewing_cutoff   = 0.25
        
        #for inn in input_noises:
        #    for outn in output_noises:
        for inn, outn in noises:
            if inn < 0.06:
                relevance_cutoff = 0.3
            elif inn < .1:
                relevance_cutoff = 0.22
            elif inn < .2:
                relevance_cutoff = 0.15
            else:
                relevance_cutoff = 0.12
            freq_ideal, filt_ideal, _, _ = compare_to_experiment(frequencies, spectra, inputNoise=inn, outputNoise=outn, verbose=False, returnFlag=True, numPoints=5000, plotFlag=False)
            fitted_rf, popt = fit_ideal(freq_ideal[freq_ideal < relevance_cutoff], filt_ideal[freq_ideal < relevance_cutoff]/np.nanmax(filt_ideal[freq_ideal < relevance_cutoff]), returnFlag='interp')
            freqs = freq_ideal[freq_ideal < viewing_cutoff]
            #if outn == 0.35:
                #plt.plot(freqs, fitted_rf(freqs, *popt), 'c', linewidth=5, alpha=0.5)
            #else:
                #plt.plot(freqs, fitted_rf(freqs, *popt), 'k', linewidth=5, alpha=0.3)
            plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color='#6699ff', linewidth=11, alpha=0.5)

            # get errorbars
            # first we need to compute the fft for all combinations of horz and amacrine cell
            horz_pfs = get_horizontal_projective_field()
            horz_pfs = get_interp(horz_pfs, mode='valid')
            ama_pfs  = get_amacrine_projective_field()
            ama_pfs  = get_interp(ama_pfs, mode='valid')

            min_space = np.max([np.max([np.min(x) for x,y in horz_pfs]), np.max([np.min(x) for x,y in ama_pfs])])
            max_space = np.min([np.min([np.max(x) for x,y in horz_pfs]), np.min([np.max(x) for x,y in ama_pfs])])
            space     = np.linspace(min_space, max_space, 100)
            rf_ffts   = []

            horz_weighting, ama_weighting, center_weighting, surround_weighting, center_width = popt

            for fh, hp in horz_pfs:
                for ah, ap in ama_pfs:
                    horz_interp = interp1d(fh, hp, kind='slinear')
                    ama_interp  = interp1d(ah, ap, kind='slinear')

                    surround = horz_weighting * horz_interp(space) + ama_weighting * ama_interp(space)

                    # make center
                    if center_width is None:
                        center = center_weighting * np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
                    else:
                        center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

                    # put them together
                    if len(center.shape) > 1:
                        center = center.squeeze()
                    rf = center_weighting * center + surround_weighting * surround

                    # Amplitude Spectrum of RF
                    two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
                    n = len(two_sided)
                    if n % 2 == 0:
                        rf_ffts.append(two_sided[:(int(n/2) + 1)])
                    else:
                        rf_ffts.append(two_sided[:(int((n-1)/2) + 1)])

            rf_ffts_err = sem(rf_ffts)
            scaling = np.nanmax(rf_f_one_sided)
            scaling_err = np.nanmax(np.mean(rf_ffts, axis=0))
            err_interp = interp1d(np.linspace(0, 1./(2.*(space[-1]-space[-2])), len(rf_ffts_err)), rf_ffts_err/scaling_err)

            #for rf_fft in rf_ffts:
            #    freqs_one_sided = np.linspace(0, 1./(2*(space[-1]-space[-2])), len(rf_fft))
            #    plt.plot(freqs_one_sided, rf_fft/scaling_err, 'c', linewidth=2)

            #if outn == 0.35:
                #plt.errorbar(freqs, fitted_rf(freqs, *popt), yerr=err_interp(freqs), color='c', alpha=0.5, linewidth=5, capthick=4, capsize=2)
            #else:
                #plt.errorbar(freqs, fitted_rf(freqs, *popt), yerr=err_interp(freqs), color='k', alpha=0.5, linewidth=5, capthick=4, capsize=2)


            
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left', 'bottom'])
        plt.ylim([0, 1.2])
        if xlimit:
            plt.xlim(xlimit)

    
    elif plotFlag == 'horizontalOnly':
        viewing_cutoff   = 0.25
        relevance_start  = 0.03
        relevance_cutoff = 0.35
        
        freq_ideal, filt_ideal, _, _ = compare_to_experiment(frequencies, spectra, inputNoise=inputNoise, outputNoise=outputNoise, verbose=False, returnFlag=True, numPoints=5000, plotFlag=False)

        mask = (freq_ideal <= relevance_cutoff) & (freq_ideal >= relevance_start)
        fitted_rf, popt = fit_ideal_horz_only(freq_ideal[mask], filt_ideal[mask]/np.nanmax(filt_ideal[mask]), returnFlag='interp')
        freqs = freq_ideal[freq_ideal < viewing_cutoff]
        plt.plot(freqs, fitted_rf(freqs, *popt), 'g', linewidth=9, alpha=0.8)
        plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color='#6699ff', linewidth=9, alpha=0.8)

        # get errorbars
        # first we need to compute the fft for all combinations of horz and amacrine cell
        horz_pfs = get_horizontal_projective_field()
        horz_pfs = get_interp(horz_pfs, mode='valid')

        min_space = np.max([np.min(x) for x,y in horz_pfs])
        max_space = np.min([np.max(x) for x,y in horz_pfs])
        space     = np.linspace(min_space, max_space, 100)
        rf_ffts   = []

        center_weighting, surround_weighting, center_width = popt

        for fh, hp in horz_pfs:
            horz_interp = interp1d(fh, hp, kind='slinear')
            surround = horz_interp(space)

            # make center
            if center_width is None:
                center = np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
            else:
                center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

            # put them together
            if len(center.shape) > 1:
                center = center.squeeze()
            rf = center_weighting * center + surround_weighting * surround

            # Amplitude Spectrum of RF
            two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
            n = len(two_sided)
            if n % 2 == 0:
                rf_ffts.append(two_sided[:(int(n/2) + 1)])
            else:
                rf_ffts.append(two_sided[:(int((n-1)/2) + 1)])

        rf_ffts_err = sem(rf_ffts)
        scaling = np.nanmax(rf_f_one_sided)
        scaling_err = np.nanmax(np.mean(rf_ffts, axis=0))
        err_interp = interp1d(np.linspace(0, 1./(2.*(space[-1]-space[-2])), len(rf_ffts_err)), rf_ffts_err/scaling_err)

        plt.errorbar(freqs, fitted_rf(freqs, *popt), yerr=err_interp(freqs), color='g', alpha=0.7, linewidth=9, capthick=5, capsize=5)
            
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left', 'bottom'])
        plt.ylim([0, 1.2])
        if xlimit:
            plt.xlim(xlimit)
        

    elif plotFlag == 'amacrineOnly':
        viewing_cutoff   = 0.25
        relevance_start  = 0.00
        relevance_cutoff = 0.3
        
        freq_ideal, filt_ideal, _, _ = compare_to_experiment(frequencies, spectra, inputNoise=inputNoise, outputNoise=outputNoise, verbose=False, returnFlag=True, numPoints=5000, plotFlag=False)

        mask = (freq_ideal <= relevance_cutoff) & (freq_ideal >= relevance_start)
        fitted_rf, popt = fit_ideal_ama_only(freq_ideal[mask], filt_ideal[mask]/np.nanmax(filt_ideal[mask]), returnFlag='interp')
        freqs = freq_ideal[freq_ideal < viewing_cutoff]
        plt.plot(freqs, fitted_rf(freqs, *popt), 'b', linewidth=9, alpha=0.8)
        plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color='#6699ff', linewidth=9, alpha=0.8)

        # get errorbars
        # first we need to compute the fft for all combinations of horz and amacrine cell
        ama_pfs = get_amacrine_projective_field()
        ama_pfs = get_interp(ama_pfs, mode='valid')

        min_space = np.max([np.min(x) for x,y in ama_pfs])
        max_space = np.min([np.max(x) for x,y in ama_pfs])
        space     = np.linspace(min_space, max_space, 100)
        rf_ffts   = []

        center_weighting, surround_weighting, center_width = popt

        for fa, ap in ama_pfs:
            ama_interp = interp1d(fa, ap, kind='slinear')
            surround = ama_interp(space)

            # make center
            if center_width is None:
                center = np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
            else:
                center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

            # put them together
            if len(center.shape) > 1:
                center = center.squeeze()
            rf = center_weighting * center + surround_weighting * surround

            # Amplitude Spectrum of RF
            two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
            n = len(two_sided)
            if n % 2 == 0:
                rf_ffts.append(two_sided[:(int(n/2) + 1)])
            else:
                rf_ffts.append(two_sided[:(int((n-1)/2) + 1)])

        rf_ffts_err = sem(rf_ffts)
        scaling = np.nanmax(rf_f_one_sided)
        scaling_err = np.nanmax(np.mean(rf_ffts, axis=0))
        err_interp = interp1d(np.linspace(0, 1./(2.*(space[-1]-space[-2])), len(rf_ffts_err)), rf_ffts_err/scaling_err)

        plt.errorbar(freqs, fitted_rf(freqs, *popt), yerr=err_interp(freqs), color='b', alpha=0.7, linewidth=9, capthick=5, capsize=5)
            
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left', 'bottom'])
        plt.ylim([0, 1.2])
        if xlimit:
            plt.xlim(xlimit)





    if statistics:
        return moreFreqs, idealFilter, rf_freqs_one_sided, rf_f_one_sided, rf_fft_f, rf_ffts
    
    if returnFlag:
        try:
            return moreFreqs, idealFilter, rf_freqs_one_sided, rf_f_one_sided
        except:
            return frequencies, idealFilter, rf_freqs_one_sided, rf_f_one_sided

def fig5_each_cell(frequencies, spectra, snr, plot_style='same'):
    ''' Generates Figure 5 with an infomax fit per ganglion cell.
    Steps:
    1) Get FFT of receptive field for each ganglion cell
    2) Fit each ganglion RF FFT to an ideal infomax curve, varying input + output noise
    3) Fit FFT of a linear combination of Ama., Horz., and Center. to the ideal infomax curve
    4) Plot all of the ideal infomax power spectra and the ffts of the models together
    
    INPUTS:
    frequencies: np array of spatial frequencies corresponding to spectra
    spectra: np array of original power spectra
    
    RETURNS:
    frequencies_ideal, ideal_filter, frequencies_expt, expt_filter
    '''

    ###### GET FFT OF GANGLION RFS #######
    ganglion_rfs  = load_ganglion_cells(micronsPerDeg=50., pca_mode='space')
    ganglion_ffts = get_fft(ganglion_rfs, mode='amplitude')

    for spatial_freq, ganglion_amp_spect in ganglion_ffts:
        # get the input and output noise corresponding to the matching ideal condition
        input_noise, output_noise = corresponding_ideal(frequencies, spectra, spatial_freq, ganglion_amp_spect, snr)

        print (input_noise, output_noise)
        
        # get the ideal filter
        freq_ideal, filt_ideal, tmp1, tmp2 = compare_to_experiment(frequencies, spectra, 
                inputNoise=abs(input_noise), outputNoise=abs(output_noise), returnFlag=True, plotFlag=False)

        model_freqs     = np.linspace(np.max([np.min(freq_ideal), np.min(spatial_freq)]),
                np.min([np.max(freq_ideal), np.max(spatial_freq)]), len(freq_ideal))
        ideal_interp    = interp1d(freq_ideal, filt_ideal, kind='slinear')
        resampled_ideal = ideal_interp(model_freqs)

        filt_model = fit_ideal(model_freqs, resampled_ideal/np.nanmax(resampled_ideal), returnFlag='array')

        if plot_style == 'random':
            r            = lambda: np.random.randint(0,255)
            random_color = '#%02X%02X%02X' % (r(),r(),r())
            plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color=random_color, alpha=0.7,
                    linestyle='.', marker='o')
            plt.plot(model_freqs, filt_model, color=random_color, alpha=0.7, linestyle='--',
                    marker='')
            plt.plot(spatial_freq, ganglion_amp_spect/np.nanmax(ganglion_amp_spect), color=random_color, 
                    alpha=0.5, linewidth=2)
        elif plot_style == 'same':
            plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color='c', alpha=0.7, linewidth=3)
            plt.plot(model_freqs, filt_model, color='b', alpha=0.7, linewidth=3)
            plt.plot(spatial_freq, ganglion_amp_spect/np.nanmax(ganglion_amp_spect), color='k', alpha=0.6, linewidth=2)

        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left', 'bottom'])
        plt.xlim([0,1])
        plt.ylim([0,1.1])
