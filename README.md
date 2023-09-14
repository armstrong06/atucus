# Arrival Time Uncertainties Created Using SWAG

From the paper **A Deep-learning Phase Picker with Calibrated Bayesian-derived Uncertainties for Earthquakes in the Yellowstone Volcanic Region** by Alysha Armstrong, Zachary Claerhout, Ben Baker, and Keith D. Koper

## Overview

Train and evaluate deep-learining P- and S-phase arrival time pickers using Stochastic Weight Averaging - Gaussian (SWAG; Maddox *et al.*, 2019) and Multiple SWAG (MultiSWAG; Wilson and Izmailov, 2020). Using SWAG, these models produce Bayesian-derived uncertainity estimates for the phase arrival times. The uncertainties are calibrated using the method from Kuleshov *et al.* (2018). The deep-learning model architecture is from Ross *et al.* (2018). 
  
[<img src="https://github.com/armstrong06/atucus/assets/46799601/8fa421a6-a737-4f62-aff4-7c912ba8481b" width="450"/>](figure8.jpeg)  
*Examples of MultiSWAG and calibration applied to two waveforms from the University of Utah Seismograph Stations (UUSS) catalog. The vertical-component waveform (Z) is shown with a histogram of predicted picks from three separate SWAG models (m1, m2, and m3), the UUSS analyst pick (y_act), the predicted pick from MultiSWAG (y_pred), the standard deviation of the MultiSWAG predictions (1 st. dev.) and the calibrated 90% credible interval (C.I.). The x-axis is relative to the analyst pick (0.0 s). (a) A P arrival labeled as high quality (0) by a seismic analyst. (b) A P arrival labeled as low quality (2) by a seismic analyst. The signal-to-noise (SNR) value for the arrival is in the top right corner.*

## Directories
- swa_gaussian-master: [SWAG code](https://github.com/wjmaddox/swa_gaussian), downloaded on May 3, 2022
- swag_modified: Modified SWAG code used in Armstrong *et al.* (2023) to produce uncertainty estimates on phase arrival time estimates
- uuss: Hyperparameter tuning, training, evaluation, and uncertainty calibration of models presented in Armstrong *et al.* (2023) 

## References
- Armstrong, A. D., Z. Claerhout, B. Baker, and K. D. Koper (2023). A deep-learning phase picker with calibrated bayesian-derived uncertainties for earthquakes in the Yellowstone Volcanic Region, Bull. Seismol. Soc. Am. XX, 1–22
- Kuleshov, V., N. Fenner, and S. Ermon (2018). Accurate uncertainties for deep learning using calibrated regression, in Proceedings of the 35th International Conference on Machine Learning Stockholm, Sweden, PMRL.
- Maddox, W., T. Garipov, P. Izmailov, D. Vetrov, and A. G. Wilson (2019). A simple baseline for Bayesian uncertainty in deep learning, Advances in Neural Information Processing Systems 32, 13153–13164.
- Ross, Z. E., M. Meier, and E. Hauksson (2018). P wave arrival picking and first‐motion polarity determination with deep learning, J. Geophys. Res. Solid Earth 123, 5120–5129.
- Wilson, A. G., and P. Izmailov (2020). Bayesian deep learning and a probabilistic perspective of generalization, in Advances in Neural Information Processing Systems, 4697–4708.
