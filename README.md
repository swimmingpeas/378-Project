# 378-Project
This is a repository for collaboration on our ELEC 378 final project


# Pre-reqs
- Due to their size, the directories 'train' and 'test' containing the respective data aren't stored in the repo and must be downloaded and put in the repository manually

# How to run the code
- The features of the model are saved in the 'features_test' and 'features_train' folders. Each set of features is stored in a .csv file. The features are extracted from the audio files in the 'test' and 'train' folders
- Since the features of the data are already extracted, the only thing necessary to interface with is the 'training_model.ipynb' file. this file contains the code to train/test the model.

# Data Exploration
- The data is stored in the 'train' and 'test' folders. Each folder contains 10 subfolders, each corresponding to a different genre of music. Each subfolder contains 100 audio files of 30 seconds each. The audio files are in .wav format.
- we began by listening to a few of the samples. Most were high quality and easy to discern the genre. However there were a few that felt as though they could belong to multiple, or even none of those listed in the assignment description. This was useful to know, as it meant that an imperfect model may be expected.
- Next, we read the files into Python in the form of numpy arrays. Checking the shape revealed that the audio files were 661,504, which is approximately 30 seconds of audio at a sample rate of 22,050 Hz. We also checked the sample rate of the audio files to confirm that it was 22,050 Hz. From this we knew that if we were going to extract features from the audio files, we would either need to reduce the size of the data by downsampling or splitting up the data, or use feature aggregation to reduce the size of the data.
- We then plotted some of the audio file's time and frequency domain representations. This was done to get a better understanding of the data and to see if there were any obvious differences between the genres in the time and frequency domains. We found that the time domain representations were very similar between the genres, but the frequency domain representations were different. This was expected, as things like percussive instruments, rhythm and tempo are more easily discerned in the frequency domain.
**MAKE PLOTS FOR ALL OF THIS STUFF**

# Feature Extraction

## Spectral Features
- These are features that are extracted through charachteristics of a sample's freuqency domain representation. They are represented as numpy arrays, where the ith element corresponds to the ith frequency band. The value of each element is the feature of the sample in that frequency band.
### centroids
- The spectral centroid indicates at which frequency the energy of a spectrum is centered upon. This is like a weighted mean:
- This is useful for determining where the "center of mass" of the spectrum is, which can determine how "bright" or "dark" a sound is.
### rolloff
- The spectral rolloff is the frequency below which some amount of the total energy of the spectrum is contained, in Librosa's case, 85%.
- Since the rolloff indicates where the majority of the energy of the spectrum is useful in analyzing where the "body" of the sound is.
### bandwidth
- The spectral bandwidth is the width of the band of frequencies in which the energy of the spectrum is concentrated.
- This is useful for determining how "sharp" or "dull" a sound is.
### contrast
- The spectral contrast is the difference in amplitude between peaks and valleys in a given frequency band.
- This indicates the "sharpness" of the sound, as higher contrast between similar frequencies indicates a sharper sound.
### flatness
- The spectral flatness is a measure of how "flat" the spectrum is. A flat spectrum has equal energy at all frequencies. For reference, white noise has a flatness of 1, while a pure tone has a flatness of 0.
- This is useful for determining how "noisy" a sound is.
### chroma stft
- The chroma stft is a 12-element vector that represents the energy of each of the 12 chroma bands.
- This is useful for determining the "color" of the sound, as it represents the energy of each of the 12 chroma bands.
### Zero crossing rate
- The zero crossing rate is the rate at which a signal changes sign.
- This is useful for determining the noisiness of a sound, as noisier sounds tend to have a higher zero crossing rate. It is also useful for determining the pitch of a sound, as the zero crossing rate is higher for higher pitched sounds.
### MFCC
- The Mel-frequency cepstral coefficients are a representation of the short-term power spectrum of a sound. They are derived from the Fourier transform of the sound.
- These are useful for determining the timbre of a sound, as they represent the power spectrum of the sound.
### Tonnetz
- The tonnetz is a 6-element vector that represents the tonal centroid features of a sound. The tonal centroid is the weighted mean of the frequencies of the sound.
- This is useful for determining the tonal characteristics of a sound, as it represents the tonal centroid features of the sound.
## Musical Features 
- These features are extracted from the time domain, and are represented as numpy arrays, where the ith element corresponds to the ith time frame. The value of each element is the feature of the sample in that time frame.
### Tempo
- Tempo is the speed at which a piece of music is played, and it's measured in BPM (beats per minute).
- This is useful, as different genres of music tend to have different tempos. For example, classical music tends to have a slower tempo, while rock music tends to have a faster tempo.
### Harmonic/Percussive
- Harmonic and percussive components are extracted from the audio signal using the Harmonic-Percussive Source Separation (HPSS) algorithm.
- This is useful for determining the harmonic and percussive components of a sound, as they are often used to distinguish between different genres of music.
### Beat Strength
- Beat strength is a measure of the strength of the beat in a piece of music. By beat stregnth, we mean the degree to which a beat is emphasized compared to the rest of the music.
- This is useful for determining the rhythm features of a sound, as different genres of music will use different time signatures more often and also make use of different rhythm patterns.

## Aggregation
- In order to reduce the size of the series data, would take up thousands of columns, we aggregated the data. We did this by taking the mean, standard deviation, and extrema of each feature. This reduced the size of the data from thousands of columns to well under a hundred, which was far more computationally efficient. In doing so, we lost some information, but felt that the tradeoff was worth it.
# Model Selection
## Neural Network

## Logistic Regression

## Random Forest

# Complete Pipeline

# Conclusions

## Models we will (should) look into
- SVM
- Logistic Regression
- Neural network
- Random Forest


# Potentially useful links
https://www.mage.ai/blog/music-genre-classification
https://www.analyticsvidhya.com/blog/2022/03/music-genre-classification-project-using-machine-learning-techniques/
https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8


# Acknowledgements:
- Members:
    - Max Kuhlman
    - Sam Lim
    - Thomas Pickell
    - Alex Zalles
    - Ben Zhao
