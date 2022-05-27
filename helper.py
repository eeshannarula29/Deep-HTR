import editdistance

with open('output.txt', 'r') as f:
    outputs = f.read().split(';')
labels = ["Imagine a vast sheet of paper on which straight", "Lines, Triangles, Squares, Pentagons, Hexagons, and other",
          "figures, instead of remaining fixed in their place,", "move freely about, on or in the surface, but without",
          "the power of rising above or sinking below it,", "very much like shadows - only hard and with," ,
          "luminars edges- and you will then have a", "pretty correct notion of my country and countrymen."]
def get_accuracy(output, label):
    error = editdistance.eval(output, label) / len(label)
    return error

total_error = 0
count = 0
for i in range(len(labels)):
    count+=1
    error = get_accuracy(outputs[i], labels[i])
    print(error)
    total_error += error
print(total_error / count)
