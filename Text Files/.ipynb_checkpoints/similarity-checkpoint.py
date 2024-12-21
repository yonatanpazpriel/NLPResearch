# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
text_1_path = "csv files/text_1.csv"  
text_2_path = "csv files/text_2.csv"  
text_3_path = "csv files/text_3.csv"  
text_4_path = "csv files/text_4.csv"  
text_5_path = "csv files/text_5.csv"  
text_6_path = "csv files/text_6.csv"  

df1, df2, df3 = pd.read_csv(text_1_path), pd.read_csv(text_2_path), pd.read_csv(text_3_path)
df4, df5, df6 = pd.read_csv(text_4_path), pd.read_csv(text_5_path), pd.read_csv(text_6_path)

data = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# Ensure the necessary column exists
if "preprocessed_text" not in data.columns:
    raise ValueError("The column 'preprocessed_text' is missing from the CSV file.")


# Vectorize the preprocessed text using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['preprocessed_text'])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Set a similarity threshold
thresholdLowerBound = 0.85  # Adjust this as needed
thresholdUpperBound = 0.9 # Adjust this as needed

# Identify file pairs with similarity above the threshold
similar_pairs = []
duplicate_files = set()
num_files = len(data)

for i in range(num_files):
    if data.loc[i, "file_name"] in duplicate_files:
        continue  # Skip files already marked as duplicates
    for j in range(num_files):
        if data.loc[j, "file_name"] in duplicate_files:
            continue  # Skip files already marked as duplicates
        similarity_score = similarity_matrix[i, j]
        if similarity_score > thresholdLowerBound:  # Similarity above threshold
            duplicate_files.add(data.loc[i, "file_name"])
            duplicate_files.add(data.loc[j, "file_name"])
            similar_pairs.add(data.loc[i, "file_name"], data.loc[j, "file_name"], similarity_score)

unique_files = num_files - len(duplicate_files)

# Display metrics
print(f"Total number of files: {num_files}")
print(f"Number of unique files: {unique_files}")
print(f"Number of files with duplicates: {len(duplicate_files)}")

# Display the results
if similar_pairs:
    print(f"There are {len(similar_pairs)} pairs of files with similarity above {thresholdLowerBound} and below {thresholdUpperBound} out of {num_files} total files:")
    for file1, file2, score in similar_pairs:
        print(f"{file1} and {file2} - Similarity: {score:.2f}")
else:
    print(f"No file pairs found with similarity above {threshold}.")

# Optional: Save the results to a CSV
output_df = pd.DataFrame(similar_pairs, columns=["File 1", "File 2", "Similarity Score"])
#output_df.to_csv("similar_files.csv", index=False) 
