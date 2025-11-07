# Ai-based-stressed-detection-using-Voice-Notes

In this project, two datasets were used for two purposes :
1. DAIC-WOZ (Clinical Context) - a widely used clinical interviews of patients for depression severity.
2. The RAVDESS (Controlled Experiment) - features 24 professional actors speaking and singing two different sentences for eight different emotions.

In the DAIC-WOZ dataset, the script aggregates 13 MFCCs, Pitch, Energy (RMS), and Speech Rate into a compact feature vector of 7 descriptive statistics per participant.
In the RAVDESS dataset,a 40-dimensional MFCC vector is extracted and The resulting feature vector is the mean of the 40 MFCC frames over that 3.0-second window.

Serveral libraries of Python were used:
1. Librosa -  The  librosa library was used as the primary means of processing and analyzing audio signals to extract major acoustic characteristics.
2. numpy - to support the computation of averages, standard deviations, and normalization of extracted features.
3. pandas -  to store the data in a DataFrame, perform feature normalization, compute combined stress scores and export the final results into a .csv file for further analysis.

 Output : Stress Score Mapping
 
 1 -> Low Stress  |
 2-> Low Moderate Stress |
 3 -> Moderate Stress |
 4 -> High Stress |
 5 -> Very High Stress
