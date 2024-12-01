import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("//home//santy//DesignProject//Medicinal_Plant.h5")

# Define the path to the image you want to test
image_path =r"/home/santy/Documents/ML Model/Indian Medicinal Leaves Image Datasets/Medicinal plant dataset/Ekka"
# Load and preprocess the image
img = image.load_img(image_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image data

# Make predictions
predictions = model.predict(img_array)

# Assuming you have a list of class labels
class_labels = ["Aloevera","Amla","Amruta_Balli","Arali","Ashoka","Ashwagandha","Avacado","Bamboo","Basale","Betel","Betel_Nut","Brahmi","Castor","Curry_Leaf","Doddapatre","Ekka","Ganike","Gauva","Geranium","Henna","Hibiscus","Honge","Insulin","Jasmine","Lemon","Lemon_grass","Mango","Mint","Nagadali","Neem","Nithyapushpa","Nooni","Pappaya","Pepper","Pomegranate","Raktachandini","Rose","Sapota","Tulasi","Wood_sorel"]


methods_of_preparation ={"slit the leaf of an aloe plant lengthwise and remove the gel from the inside, or use a commercial preparation./n"
    ,"Eating raw amla and candies or taking amla powder with lukewarm water/n"
    ,"Make a decoction or powder from the stems of Giloy. It is known for its immunomodulatory properties./n"
    ," Various parts like the root bark, leaves, and fruit are used for medicinal purposes. It can be consumed in different forms, including as a decoction.","Different parts like the bark are used. It's often prepared as a decoction for menstrual and uterine health.","The root is commonly used, and it can be consumed as a powder, capsule, or as a decoction. It is an adaptogen known for its stress-relieving properties.","The fruit is consumed for its nutritional benefits, including healthy fats and vitamins./n"
    ,"Bamboo shoots are consumed, and some varieties are used in traditional medicine.","The leaves are consumed as a leafy vegetable. It's rich in vitamins and minerals."," Chewing betel leaves with areca nut is a common practice in some cultures. It's believed to have digestive and stimulant properties.","The nut is often chewed with betel leaves. However, excessive consumption is associated with health risks./n"
    ,"The leaves are used for enhancing cognitive function. It can be consumed as a powder, in capsules, or as a fresh juice.","Castor oil is extracted from the seeds and used for various medicinal and cosmetic purposes.","Curry leaves are used in cooking for flavor, and they are also consumed for their potential health benefits.","The leaves are used in traditional medicine, often as a poultice for skin conditions./n"
    ,"Various parts may be used in traditional medicine. It's important to note that some species of Ekka may have toxic components, and proper identification is crucial.","The leaves are used in traditional medicine, often as a remedy for respiratory issues.","Guava fruit is consumed for its high vitamin C content and other health benefits.","Geranium oil is extracted from the leaves and stems and is used in aromatherapy and skincare./n"
    ," Henna leaves are dried and powdered to make a paste used for hair coloring and as a natural dye.","Hibiscus flowers are commonly used to make teas, infusions, or extracts. They are rich in antioxidants and can be beneficial for skin and hair health.","Various parts of the tree are used traditionally, including the bark and seeds. It's often used for its anti-inflammatory properties.","The leaves are used for their potential blood sugar-lowering properties. They can be consumed fresh or as a tea.","Jasmine flowers are often used to make aromatic teas or essential oils, known for their calming effects./n"
    ,"Lemon juice is a common remedy for digestive issues, and the fruit is rich in vitamin C. The peel can be used for its essential oil.","Lemon grass is used to make teas and infusions, known for its soothing and digestive properties.","Mango fruit is consumed fresh and is rich in vitamins and minerals. Some parts, like the leaves, are also used in traditional medicine.","Mint leaves are commonly used to make teas, infusions, or added to dishes for flavor. It's known for its digestive properties.","Different parts of the plant are used traditionally. It's often prepared as a decoction./n"
    ,"Various parts of the neem tree are used, including leaves, bark, and oil. It's known for its antibacterial and antifungal properties.","The flowers are used in traditional medicine, often for their calming effects."," Different parts of the tree are used traditionally. The oil extracted from the seeds is used for various purposes./n"
    ," Different parts of the tree are used traditionally. The oil extracted from the seeds is used for various purposes.","Spice for flavor; potential digestive and antimicrobial properties."," Eat seeds or drink juice for antioxidant benefits.","Traditional uses; some parts may be toxic, use caution."," Make tea or use petals for calming and aromatic effects./n"
    ,"Consume fruit for its sweet taste and nutritional content."," Make tea or use leaves for immune support.","Use leaves in salads; some varieties contain oxalic acid"}


use_of_medicine =[" {improve skin and prevent wrinkles,wound healing}","{ controlling diabetes,hair amazing,losing weight,skin healthy}","{Immunomodulatory, fever.}","{ parts for traditional healing.}/n"
    ,"{Uterine health, menstrual issue.}","{Adaptogen, stress relief.}","{ Nutrient-rich, heart health.}","{Shoots, traditional cuisine.}","{Shoots, traditional cuisine.}","{Digestive, chewed with areca nut.}","{Chewing, traditional practices, caution.}","{Cognitive enhancer, adaptogen}","{ Oil for medicinal, cosmetic use}/n"
    ,"{ Flavoring, potential traditional uses.}","{ Poultice, skin conditions.}","{Traditional uses, caution for toxicity.}","{Respiratory health, traditional medicine.}","{ Vitamin C, digestive benefits}","{ Oil for aromatherapy, skincare.}","{ Hair coloring, natural dye.}/n"
    ,"{Tea for antioxidants, skin health.}","{Anti-inflammatory, traditional use.}","{Potential blood sugar regulation, traditional use.}","{Tea, relaxation, stress relief.}","{Digestive aid, rich in vitamin C.}/n"
    ,"{Tea, digestive, calming effects.}","{Fruit, traditional uses for health.}","{Tea, aids digestion, refreshing flavor.}","{Traditional uses, potential medicinal purposes.}","{ Antibacterial, antifungal, supports skin health.}/n"
    ,"{Calming effects, traditional use.}","{ Oil from seeds, various traditional uses.}","{ Fruit, leaves, traditional uses.}","{ Spice, potential digestive benefits.}","{Antioxidant-rich, heart health.}/n"
    ,"{Traditional uses, caution for potential toxicity.}","{Tea, calming, aromatic effects.}","{Sweet taste, nutritional content.}","{Tea, immune support, respiratory health.}","{Leaves in salads, some varieties may have medicinal uses.}"]

# Get the predicted class index
predicted_class_index = np.argmax(predictions)#index value

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]
predicted_labels_methods= methods_of_preparation[predicted_class_index]
medicine_use= use_of_medicine[predicted_class_index]

# Display the results
print("Predicted class :", predicted_class_label)
print("method of preparation of ", predicted_class_label," : ", predicted_labels_methods)
print("Predictions:", predictions)
print("use of the medicine :", medicine_use)