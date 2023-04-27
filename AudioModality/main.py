from sklearn.model_selection import train_test_split
import os
import numpy

#Step 1: Split training and testing sets
angry_path = 'angry'
angry_files = os.listdir(angry_path)
angry_train, angry_test = train_test_split(angry_files, test_size=.3,random_state=42)

print(f'Number of training files: {len(angry_train)}')
print(f'Number of testing files: {len(angry_test)}')