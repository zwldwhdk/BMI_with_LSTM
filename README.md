# BMI_with_LSTM
Brain Machine Interface with Deep Neural Network, Long Short Term Memeory

# Datasets
You can find specific details of processing at Yeom et al., (2012)., Journal of Neuroengineering.
1) Brain data

  Non-invasive Brain signal data; especially EEG or MEG
  For this cords, I used 306 channels MEG data(204 gradiometers out of whole channel exactly) which was collected from Yeom et al., (2012) experiment.
For a decoder, which signal preprocessing is needed, .5 to 8 Hz width Bandpass filtering is applied, and downsampled through time steps. For the ohters which preprocessing does not needed, only normalization is applied through time steps.  
2) Hand Trajectory
3D Hand movement Trajectory is low band filtered, and down sampled through time.

# Decoder
*I uploaded 3 decoder types for MEG signal decoding.
1) single-layered bidirectional Long-Short Term Memory(bLSTM)

The data is preprocessed with Bandpass filtering [.8 to 5] bandwidth,

2) 3 layered Convolution Neural Network
3) RCNN; 3-layered CNN + single-layer bLSTM
