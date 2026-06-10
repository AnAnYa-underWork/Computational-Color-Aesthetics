import csv
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from colorsys import rgb_to_hsv


# Example data entry (replace with your actual data processing)
data = [
        {
            "aesthetic": "Vintage",
            "palette": ["#8C6E58", "#B79A79", "#D3C2AD", "#F3E6D1", "#DEC4A1"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#7A5A3E", "#C5A57D", "#D9C2A6", "#F5EAD4", "#A47B55"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4E3629", "#7D5A4F", "#B89E84", "#DEC9A5", "#F5E4C8"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6A4534", "#B6805F", "#D4A57E", "#F1D5B5", "#A05C42"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#56443D", "#9B8071", "#C8B29D", "#E9D9C5", "#AF937A"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#3E2A25", "#6E4B41", "#B1927B", "#E4CDB3", "#9D7A66"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#785E4F", "#AF8C76", "#D2BDA1", "#F7E6D5", "#CEA381"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#5A4338", "#9C7259", "#BFA48A", "#F3DEC8", "#836548"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#452E2A", "#744E47", "#A98C7B", "#DBC9B3", "#7F6057"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6E513D", "#B6936A", "#D5B793", "#F9E8D4", "#A07053"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#704C40", "#9E7D6D", "#C3A591", "#E9D8BE", "#AD8E7A"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#55382F", "#8F6A5C", "#C5A58E", "#E5D1BB", "#9C7A66"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4A382E", "#7A5A49", "#B29274", "#E2C5A5", "#8B705E"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#664538", "#A17B62", "#C7A48A", "#EBD3B6", "#8E6147"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6A4B3A", "#A07E69", "#D2BBA2", "#F4E2CC", "#C6937F"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4D2F29", "#7B574D", "#B78F7A", "#E4C5AA", "#9A7266"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#735944", "#A7856C", "#CFB292", "#F6E1C9", "#8D6B51"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#563B2E", "#8D6C5A", "#C19A7D", "#E7CEB0", "#9A745D"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6F4F3A", "#A37F63", "#D0B28C", "#F3E2C8", "#B58F71"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#674B39", "#A07B60", "#C7A083", "#F2D6B7", "#936B56"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#553A32", "#8C6754", "#BA957C", "#DFC5A5", "#7E5A4C"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#623D31", "#9B6F58", "#C8A08A", "#EDD4BE", "#855945"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#7A503F", "#B3886F", "#D3B399", "#F8E1CA", "#A06F56"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4B362D", "#7E5C48", "#B08A6F", "#DBC6A9", "#8C6752"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#8C614A", "#B78C6F", "#DFC4A4", "#F5E2CB", "#AA7C5E"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#5C3F33", "#966F58", "#BDA285", "#EDD3BB", "#8B6A53"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#755139", "#A68461", "#C9A883", "#F4D6B9", "#A26C51"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#5E4132", "#967259", "#BE9F7D", "#EAD1B2", "#81644B"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#65412F", "#A17A5F", "#C9A486", "#F1D7BC", "#8A684D"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#472E27", "#7B564A", "#A78D76", "#D5BCA3", "#805C4D"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6C4938", "#9C7D6A", "#C2A692", "#EFD7C0", "#8A6554"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4F362B", "#7A5B4B", "#A88D77", "#D9C5AC", "#8C6A5B"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#5E4333", "#94745C", "#C5A384", "#EAD6B7", "#7F5A47"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#714D3A", "#A18068", "#CBAF8C", "#F3DEC4", "#906649"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#674939", "#A0785E", "#C6A98A", "#F0D5BA", "#8A624B"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#5B3D2F", "#8D6B54", "#B89779", "#E2C6A6", "#7A5A46"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#755944", "#A5856B", "#CEAE8E", "#F7DCC4", "#9B7259"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6F4A37", "#A17A63", "#D1AE8C", "#F6DAC4", "#8E654F"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4E392E", "#815E49", "#B39275", "#E3C6A8", "#6F5648"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6A4938", "#96745C", "#C5A183", "#EFD3B7", "#8A5E47"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#5C4031", "#886651", "#B4977B", "#E2C4A4", "#7E5944"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#714C3A", "#A67F65", "#D4AF8D", "#F8DEC7", "#92654D"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4F3A2D", "#7D5E49", "#B29074", "#E3C5A8", "#6F5544"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#5A3D30", "#8A6B54", "#B89577", "#E1C5A4", "#7D5A45"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#724F3A", "#A67F66", "#D3AF8C", "#F8DEC6", "#93654C"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#664638", "#98785D", "#C6A384", "#EED3B6", "#835949"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#4E382D", "#7F5F49", "#B38F75", "#E3C6A9", "#6F5545"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6B493A", "#99765E", "#C7A186", "#F0D6BA", "#8C5E49"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#553A2F", "#8A6B54", "#B89577", "#E1C5A4", "#7E5945"]
        },
        {
            "aesthetic": "Vintage",
            "palette": ["#6F4A36", "#A17A62", "#D1AD8C", "#F4DAC4", "#8F654E"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F2D7D5", "#FAD4C0", "#FDE2E4", "#D8E2DC", "#FFE5D9"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F8E9A1", "#F76C6C", "#A8D0DB", "#374785", "#24305E"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#C3B1E1", "#EAC7C7", "#FFE5E2", "#B9FBC0", "#FFDDC1"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFCFDF", "#C7CEEA", "#FDFD96", "#C6E2FF", "#FFABAB"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#E7E6E1", "#A8D5BA", "#F9EBC8", "#F4A8A8", "#95C8D8"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#AFCBFF", "#C4FAF8", "#F6EAC2", "#FFA8A8", "#FFDCDC"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#E5D9F2", "#D4A5A5", "#F9D8D8", "#FAF4B7", "#98DDCA"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#C4FCEF", "#FFD8BE", "#FFABAB", "#FFDAF9", "#FBE7C6"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFE5E2", "#FFCBCB", "#EFD3D7", "#F3D4C3", "#C8E4D0"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FAD9C1", "#F5C7A9", "#FFB6B9", "#A9DEE4", "#9DCC8C"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFDDC1", "#FFABAB", "#FFC3A0", "#D4E09B", "#F6A6B2"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFE0B2", "#FFAB76", "#FF6768", "#A4EBF3", "#90E0EF"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFDAC1", "#FFB7B2", "#FF9AA2", "#FFB7B2", "#C7CEEA"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F7F4EA", "#B6D7A8", "#F1C0E8", "#E8E8A6", "#A2C4C9"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FDE2E4", "#FAD2E1", "#F9C6C9", "#E2ECE9", "#B8DE6F"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#D3B8AE", "#F4D9D8", "#FAE3E3", "#FBE9E7", "#F4C8C7"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FDE8D6", "#FFD7BA", "#FEC89A", "#FFA69E", "#F6BD60"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFE7D6", "#FFA8A8", "#FFD1BA", "#B9FBC0", "#FAF0CA"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#C5CBE3", "#DFCFEA", "#F7D9C4", "#FBE7C6", "#FFF9B0"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFECDF", "#F8D5B8", "#FFAAA7", "#FFEEDB", "#C8D5B9"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFE5EC", "#FFC2D1", "#FFB3C6", "#FF999F", "#FFB0B0"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFC6C7", "#FEE9E2", "#FAF3DD", "#D4E4BC", "#FFF1E3"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFCBD2", "#FFF8E8", "#FFEBB7", "#D6F5E5", "#C6DBE3"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#E7E8D1", "#D3E2DD", "#D6E1E3", "#E1C6CA", "#F6C3C3"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F4D7C3", "#D2E7D3", "#C6E4DA", "#D7F9FF", "#FFE7D1"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#E1F4F3", "#B8D6C7", "#F1C0E8", "#E9E9D8", "#FFDFDE"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFC2E9", "#FCE6E8", "#F3D1D1", "#E7F4F2", "#FFDFD3"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FCEADE", "#FFE6C6", "#FBEAD3", "#FFD3CA", "#F8AFA6"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFE7DE", "#FAE3E3", "#FFC5C5", "#D8E5E1", "#C7E4C4"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#D4D9D1", "#E5E5CF", "#CDEAC0", "#F6E8D4", "#F8D7CD"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FAD4C0", "#FFE6C6", "#E8D4C4", "#F8E8D3", "#FAD9D6"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFDCDC", "#FFE6DB", "#FFDEDE", "#FEE5E2", "#FBD5D5"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFEBE7", "#E7E7D1", "#FAF3DD", "#FFF8E8", "#E8D8D7"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFF7EB", "#FCE5E5", "#FFD8C9", "#F9D4D4", "#F2E9E4"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFF5DA", "#F7E3E3", "#FFE6D9", "#FFC7C4", "#FFC5A8"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFE4C1", "#FFD4CC", "#FCE5D8", "#F8D7CD", "#EDE8C3"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FAD9E8", "#FFEBE8", "#FFC4C4", "#D9E8F6", "#FFE3C4"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#E8E6E3", "#F8D4D4", "#F9E6DD", "#D4E5DD", "#C8E5D7"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F8D9DA", "#FFC7C7", "#FFE1D1", "#E9E3CF", "#C7DECF"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFDAD9", "#EDE3D4", "#FAE5E3", "#C7E3D3", "#D5D9F6"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F6C8E3", "#FFD8DA", "#F9E3C6", "#E9C9C9", "#F4D6D6"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FBEAD6", "#F6C5C5", "#FFC1C1", "#E3E3CF", "#E6E6E3"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFDFDD", "#FCE1D9", "#E4D3D3", "#F1C8C9", "#FAE6E3"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F7D4D3", "#E9E8DF", "#FFDAD8", "#C5E2C9", "#FFCCC1"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFF3E3", "#FBE7E8", "#E3E6DF", "#E8C6C4", "#F7D3C6"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FEE0C8", "#EFD3D4", "#FFCCC9", "#F8D7D6", "#C8E3E2"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F8D8D7", "#FCE7D4", "#E7E8E6", "#FAD4D6", "#D6E6D8"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#FFD7D7", "#FFF4E0", "#D9E4DD", "#C8D8E4", "#E8D1E1"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#F7E4C8", "#FFE6E6", "#D4E5F0", "#C5DDE8", "#F2E8D9"]
        },
        {
            "aesthetic": "Pastel",
            "palette": ["#E6E6F2", "#C8D8E4", "#FAD4C4", "#FFD8E4", "#DDEBF0"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#C2B8A3", "#A17C6B", "#D4A59A", "#F5E4C3", "#705D56"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#946F43", "#E0C097", "#D9CAB3", "#B7A49A", "#7B574D"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D1A28F", "#B2857F", "#F4CBA1", "#C19A86", "#8C6450"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D8B4A0", "#B27C7D", "#F1D8B2", "#84625D", "#6F493F"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#6A514A", "#9B776D", "#D6AD8C", "#F5E4C3", "#A19485"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#B7A495", "#D8CBBE", "#D4A59A", "#8C6450", "#715757"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#BF9474", "#D6C8B2", "#A37967", "#8C7E72", "#5E504A"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D8BEA1", "#9F857B", "#805F50", "#C2AB9A", "#EBE1D5"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#C8A982", "#856D58", "#C2AB97", "#D5C3B0", "#F2E6D9"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#B28973", "#D5C3B0", "#9F857A", "#B29E88", "#F4D8B2"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#735548", "#D4A59A", "#E5CEBD", "#A17C6B", "#6A514A"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D1B7A1", "#A17C6B", "#735548", "#C8A982", "#F5E4C3"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#B2857F", "#6F493F", "#D6AD8C", "#E1CBB7", "#9B776D"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#A68F79", "#F2E6D9", "#C2AB9A", "#BF9474", "#6E4A34"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D4A59A", "#B2857F", "#C19A86", "#9F857B", "#84625D"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#C8A982", "#D5C3B0", "#A17C6B", "#805F50", "#F1D8B2"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#9B776D", "#B7A495", "#C2B8A3", "#D1A28F", "#6A514A"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#F2E6D9", "#C8A982", "#705D56", "#D6AD8C", "#84625D"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#9F857A", "#C2AB97", "#A68F79", "#F4D8B2", "#735548"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D4A59A", "#6E4A34", "#B7A495", "#E5CEBD", "#D6C8B2"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#84625D", "#B28973", "#F5E4C3", "#C8A982", "#735548"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#A68F79", "#705D56", "#C2B8A3", "#D1A28F", "#E5CEBD"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D8BEA1", "#9B776D", "#6F493F", "#D4A59A", "#B28973"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#A17C6B", "#B2857F", "#F2E6D9", "#9F857B", "#84625D"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#B7A495", "#C2AB97", "#D1B7A1", "#A68F79", "#6E4A34"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#6A514A", "#D6AD8C", "#E5CEBD", "#D4A59A", "#735548"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D6C8B2", "#F1D8B2", "#B28973", "#9F857A", "#705D56"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#B2857F", "#D4A59A", "#C8A982", "#D1A28F", "#6F493F"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#F4D8B2", "#C2AB9A", "#A17C6B", "#705D56", "#E5CEBD"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D1B7A1", "#D6AD8C", "#F5E4C3", "#9B776D", "#6A514A"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#705D56", "#B2857F", "#84625D", "#F1D8B2", "#A17C6B"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D8BEA1", "#A68F79", "#735548", "#D4A59A", "#C8A982"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D6AD8C", "#B28973", "#9F857A", "#6F493F", "#C2AB97"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#C2AB97", "#D5C3B0", "#D1A28F", "#705D56", "#A17C6B"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#6E4A34", "#B7A495", "#D4A59A", "#B2857F", "#F2E6D9"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#9B776D", "#D8BEA1", "#A68F79", "#C2B8A3", "#735548"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#F5E4C3", "#D4A59A", "#C8A982", "#9F857B", "#6A514A"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#84625D", "#D6AD8C", "#B28973", "#A17C6B", "#F4D8B2"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#D6C8B2", "#E5CEBD", "#705D56", "#B2857F", "#6F493F"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#B7A495", "#D8BEA1", "#D1A28F", "#735548", "#6A514A"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#C8A982", "#A68F79", "#D4A59A", "#B28973", "#9F857A"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#F2E6D9", "#9B776D", "#6E4A34", "#D5C3B0", "#C2AB97"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#705D56", "#C2B8A3", "#A17C6B", "#D4A59A", "#D6AD8C"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#C8A982", "#D1A28F", "#D5C3B0", "#6F493F", "#9F857B"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#F5E4C3", "#84625D", "#B2857F", "#D4A59A", "#A68F79"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#B7A495", "#D6AD8C", "#705D56", "#F1D8B2", "#C2AB97"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#6E4A34", "#D1B7A1", "#D8BEA1", "#B28973", "#735548"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#C4A69D", "#F1D9C6", "#927B6B", "#6F5B53", "#EAD4C1"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#BCA690", "#D8C2AE", "#806858", "#F2DEC4", "#9F8576"]
        },
        {
            "aesthetic": "Bohemian",
            "palette": ["#A88C7A", "#D1B6A2", "#6D5548", "#C7A68F", "#F5E3D3"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF00FF", "#00FFFF", "#FF4500", "#8A2BE2", "#7FFF00"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#39FF14", "#FF1493", "#1E90FF", "#FFD700", "#FF69B4"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FF00", "#FF007F", "#9400D3", "#FF6347", "#00BFFF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF0000", "#7CFC00", "#00FA9A", "#8A2BE2", "#FFD700"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FFFF", "#FF1493", "#ADFF2F", "#FF4500", "#FF00FF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#32CD32", "#FF69B4", "#1E90FF", "#FFD700", "#9400D3"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF6347", "#8A2BE2", "#00BFFF", "#FF1493", "#FF00FF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#39FF14", "#00FA9A", "#FF4500", "#9400D3", "#FFD700"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF1493", "#7CFC00", "#00FFFF", "#8A2BE2", "#FF007F"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FF00", "#FF00FF", "#ADFF2F", "#FF69B4", "#FFD700"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#32CD32", "#9400D3", "#FF4500", "#00FFFF", "#FF6347"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF1493", "#1E90FF", "#FF007F", "#39FF14", "#FF00FF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FA9A", "#8A2BE2", "#FF4500", "#FF6347", "#00FFFF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#7FFF00", "#FF00FF", "#FFD700", "#9400D3", "#FF007F"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF1493", "#00FF00", "#00BFFF", "#FF6347", "#8A2BE2"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#39FF14", "#1E90FF", "#FF00FF", "#FF4500", "#FF69B4"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#ADFF2F", "#FF007F", "#9400D3", "#00FFFF", "#FFD700"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF6347", "#32CD32", "#FF00FF", "#8A2BE2", "#FF69B4"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FF00", "#FF1493", "#FFD700", "#00BFFF", "#7FFF00"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF4500", "#8A2BE2", "#FF007F", "#1E90FF", "#00FA9A"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF00FF", "#39FF14", "#9400D3", "#FF6347", "#ADFF2F"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#7FFF00", "#FF007F", "#00FFFF", "#FF4500", "#FF1493"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FF00", "#1E90FF", "#FF6347", "#9400D3", "#FF69B4"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF1493", "#ADFF2F", "#FF00FF", "#8A2BE2", "#00FA9A"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#39FF14", "#00FFFF", "#9400D3", "#FF69B4", "#FF007F"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF4500", "#FF6347", "#FFD700", "#32CD32", "#00BFFF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#8A2BE2", "#00FFFF", "#FF00FF", "#7FFF00", "#FF1493"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FA9A", "#FF69B4", "#1E90FF", "#ADFF2F", "#9400D3"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF4500", "#FF6347", "#00FFFF", "#FF007F", "#39FF14"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF00FF", "#ADFF2F", "#7FFF00", "#FF1493", "#9400D3"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF6347", "#00FFFF", "#1E90FF", "#32CD32", "#FF4500"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#8A2BE2", "#FF1493", "#FF00FF", "#00BFFF", "#00FA9A"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF69B4", "#39FF14", "#FFD700", "#9400D3", "#7FFF00"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FFFF", "#FF4500", "#FF00FF", "#8A2BE2", "#1E90FF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF6347", "#ADFF2F", "#00FA9A", "#FF1493", "#9400D3"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#7FFF00", "#FF007F", "#32CD32", "#FFD700", "#00BFFF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF1493", "#00FFFF", "#FF4500", "#FF69B4", "#39FF14"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#8A2BE2", "#FF00FF", "#9400D3", "#7FFF00", "#1E90FF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF007F", "#ADFF2F", "#00FA9A", "#FF6347", "#FF4500"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF6347", "#00FFFF", "#FF1493", "#FF007F", "#FFD700"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#9400D3", "#32CD32", "#8A2BE2", "#FF00FF", "#7FFF00"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF69B4", "#1E90FF", "#00FA9A", "#39FF14", "#ADFF2F"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF4500", "#7FFF00", "#00FFFF", "#9400D3", "#FF1493"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF00FF", "#ADFF2F", "#FF6347", "#1E90FF", "#00FA9A"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#00FFFF", "#FF69B4", "#9400D3", "#FF007F", "#8A2BE2"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#32CD32", "#FFD700", "#39FF14", "#FF1493", "#7FFF00"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#ADFF2F", "#00FA9A", "#FF4500", "#FF00FF", "#00BFFF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF6347", "#8A2BE2", "#FF69B4", "#9400D3", "#1E90FF"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#FF007F", "#00FF00", "#FFD700", "#1E90FF", "#FF4500"]
        },
        {
            "aesthetic": "Neon",
            "palette": ["#9400D3", "#FF1493", "#39FF14", "#00FFFF", "#FF6347"]
        },
        { "aesthetic": "Dark Academia",
    "palette": ["#2C2C2C", "#4E4E50", "#6E675F", "#8A7968", "#A8998B"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#3A3A3A", "#524A4E", "#6C757D", "#9A8C98", "#D1C7BD"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1E1E24", "#474747", "#726A61", "#B3A394", "#E4DED7"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2B2B28", "#4F4B3A", "#716953", "#A89F91", "#D5D1CB"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#292724", "#5A5353", "#8B7F7F", "#BFB8B1", "#E9E4DF"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1C1C1A", "#3B3B37", "#7B7B74", "#B9B7A7", "#EFEAE2"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#191919", "#333333", "#5D5D5D", "#A9A9A9", "#DADADA"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2B2A27", "#4E4C4B", "#716A67", "#B5B1AC", "#E7E3DF"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#312F2F", "#524E4D", "#756F6C", "#A8A099", "#D8D0C8"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#272422", "#48443E", "#70675F", "#A19688", "#E6E0D9"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2E2B2A", "#504947", "#746F6B", "#A8A49E", "#E2DED7"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2C2825", "#4F4741", "#72675F", "#A29C8C", "#E4DED3"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1F1D1B", "#43403C", "#6C6763", "#9D9792", "#CCC9C5"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#272423", "#4B4744", "#7C756F", "#AEA99F", "#D6D3C9"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#231F20", "#4A4341", "#746C69", "#AFA49E", "#E1D8D4"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2A2826", "#534E4B", "#7A736F", "#B1ABA6", "#D9D4D1"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#292725", "#504B47", "#79716D", "#ACA79F", "#DAD4CE"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#212121", "#464646", "#737373", "#A9A9A9", "#D5D5D5"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1E1E1E", "#3D3D3D", "#666666", "#B3B3B3", "#E1E1E1"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#252525", "#424242", "#6E6E6E", "#B1B1B1", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#191919", "#333333", "#4F4F4F", "#B3B3B3", "#E1E1E1"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#282828", "#4A4A4A", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#292929", "#474747", "#6E6E6E", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#232323", "#4E4E4E", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#202020", "#424242", "#6E6E6E", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2C2C2C", "#4E4E4E", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1F1F1F", "#3D3D3D", "#666666", "#B3B3B3", "#E1E1E1"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#242424", "#464646", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#292929", "#4A4A4A", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#252525", "#424242", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#282828", "#474747", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2F2F2F", "#4F4F4F", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#252525", "#444444", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#2A2A2A", "#474747", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1E1E1E", "#424242", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#262626", "#4B4B4B", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1F1F1F", "#444444", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#292929", "#484848", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#242424", "#464646", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#262626", "#4B4B4B", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#232323", "#424242", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#1F1F1F", "#444444", "#737373", "#B3B3B3", "#E6E6E6"]
    },
    {
        "aesthetic": "Dark Academia",
        "palette": ["#252525", "#4F4F4F", "#737373", "#B3B3B3", "#E6E6E6"]
    },
{
    "aesthetic": "Dark Academia",
    "palette": ["#3D2B1F", "#5A4736", "#8C756A", "#B8A49B", "#DED6C5"]
},
{
    "aesthetic": "Dark Academia",
    "palette": ["#2A2827", "#4B4846", "#726E6A", "#A59E97", "#D7D0C9"]
},
{
    "aesthetic": "Dark Academia",
    "palette": ["#2C1F1F", "#503D34", "#7A6259", "#A99085", "#D9C7BD"]
},
{
    "aesthetic": "Dark Academia",
    "palette": ["#23211F", "#46433F", "#716C67", "#A29E96", "#D7D2CA"]
},
{
    "aesthetic": "Dark Academia",
    "palette": ["#3B3029", "#5C4F44", "#877266", "#B19D8E", "#DED3C4"]
},
{
    "aesthetic": "Dark Academia",
    "palette": ["#2E261F", "#4F4337", "#7D6959", "#AC9A87", "#DFD2BF"]
}
]

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

def calculate_luminance(rgb):
    """Calculate luminance for a single RGB color."""
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def calculate_saturation(rgb):
    """Calculate saturation for a single RGB color."""
    r, g, b = rgb
    max_rgb = max(r, g, b)
    min_rgb = min(r, g, b)
    return (max_rgb - min_rgb) / max_rgb if max_rgb != 0 else 0

def rgb_to_hue(r, g, b):
    """Convert RGB to Hue in degrees."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, _, _ = rgb_to_hsv(r, g, b)
    return h * 360  # Convert hue to degrees (0-360)

def calculate_contrast(colors_rgb):
    """Calculate contrast based on luminance values."""
    luminance_values = [calculate_luminance(color) for color in colors_rgb]
    return max(luminance_values) - min(luminance_values)

# Save data and extract features
with open("updated_color_palettes.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    header = ["Aesthetic", "Mean_R", "Mean_G", "Mean_B", "Std_R", "Std_G", "Std_B",
              "Mean_Luminance", "Mean_Saturation", "Contrast"]
    for i in range(1, 6):  # Assuming 5 colors per palette
        header += [f"Color{i}_R", f"Color{i}_G", f"Color{i}_B", f"Color{i}_Hue"]
    writer.writerow(header)

    # Process each palette
    for entry in data:
        aesthetic = entry["aesthetic"]
        colors_rgb = np.array([hex_to_rgb(color) for color in entry["palette"]])

        # Compute basic statistics
        mean_color = colors_rgb.mean(axis=0)
        std_dev = colors_rgb.std(axis=0)

        # Compute additional features
        luminance_values = [calculate_luminance(color) for color in colors_rgb]
        mean_luminance = np.mean(luminance_values)
        saturation_values = [calculate_saturation(color) for color in colors_rgb]
        mean_saturation = np.mean(saturation_values)
        contrast = calculate_contrast(colors_rgb)
        hues = [rgb_to_hue(*color) for color in colors_rgb]

        # Flatten RGB and Hue values
        flattened_colors = colors_rgb.flatten()
        flattened_hues = hues

        # Write row to CSV
        row = [aesthetic] + list(mean_color) + list(std_dev) + [mean_luminance, mean_saturation, contrast] + \
              list(flattened_colors) + list(flattened_hues)
        writer.writerow(row)

# Reload the updated dataset
data_df = pd.read_csv("updated_color_palettes.csv")

# Define the expected features explicitly
features = [
    "Mean_R", "Mean_G", "Mean_B", "Std_R", "Std_G", "Std_B",
    "Mean_Luminance", "Mean_Saturation", "Contrast"
]
for i in range(1, 6):  # Assuming 5 colors per palette
    features += [f"Color{i}_R", f"Color{i}_G", f"Color{i}_B", f"Color{i}_Hue"]

# Ensure the dataset has the required columns
missing_features = set(features) - set(data_df.columns)
if missing_features:
    raise ValueError(f"The following features are missing in the dataset: {missing_features}")

# Extract features (X) and target (y)
X = data_df[features]
y = data_df["Aesthetic"]

# Debug: Check the number of features
print(f"Features used for training: {list(X.columns)}")
print(f"Number of features in training set: {X.shape[1]}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(svm_model, "updated_svm_model.pkl")

# Debug: Confirm test set has the same features
print(f"Features in test set: {list(X_test.columns)}")
print(f"Number of features in test set: {X_test.shape[1]}")

# Evaluate the model
y_pred = svm_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predictions on new data (if applicable)
new_data_df = pd.read_csv("updated_color_palettes.csv")  # Example: load new palettes
if not set(features).issubset(new_data_df.columns):
    raise ValueError("New data does not contain all required features for prediction.")

X_new = new_data_df[features]  # Use the same features as training
y_pred_new = svm_model.predict(X_new)
print("Predictions on New Data:", y_pred_new)



# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=svm_model.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# 2. Feature Importance (Linear Kernel SVM)
if svm_model.kernel == "linear":
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Coefficient": svm_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="Coefficient", y="Feature", palette="coolwarm")
    plt.title("Feature Importance (Linear Kernel SVM)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# 3. PCA Projection of Data for Visualization
pca = PCA(n_components=2)  # Reduce to 2D for visualization
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot training data in PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette="Set2", s=50)
plt.title("Training Data in PCA Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Ensure Seaborn aesthetic styles
sns.set(style="whitegrid")

# Define constants for debugging and safety
DEBUG = True  # Set to False to suppress debugging prints

# Helper function for debugging
def debug_print(message, data=None):
    if DEBUG:
        print(message)
        if data is not None:
            print(data)
            print("-" * 80)

# Assuming `svm_model`, `X_train`, `y_train`, and `pca` are already defined and trained
try:
    # Perform PCA on the training data (ensure `pca` is already fitted)
    X_train_pca = pca.transform(X_train)
    debug_print("Transformed PCA data (sample)", X_train_pca[:5])

    # STEP 1: Create a grid of points in the PCA space
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

    debug_print("PCA Component Ranges", {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})

    # Define grid resolution and bounds
    step_size = 1  # Use a larger step size to avoid memory issues
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    debug_print("Grid shape (xx, yy)", {"xx_shape": xx.shape, "yy_shape": yy.shape})

    # Flatten the grid and transform back to the original feature space
    grid_points_pca_space = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original_space = pca.inverse_transform(grid_points_pca_space)

    debug_print("Grid points in PCA space (sample)", grid_points_pca_space[:5])
    debug_print("Grid points in original space (sample)", grid_points_original_space[:5])

    # STEP 2: Predict on the grid points
    try:
        Z = svm_model.predict(grid_points_original_space)
    except Exception as e:
        debug_print("Error during SVM prediction", str(e))
        raise

    debug_print("Predicted values (Z sample)", Z[:10])

    # Convert string labels to numerical values
    label_encoder = LabelEncoder()
    Z_numeric = label_encoder.fit_transform(Z)

    debug_print("Numerical predictions (Z_numeric sample)", Z_numeric[:10])

    # Reshape predictions to match grid
    Z_numeric = Z_numeric.reshape(xx.shape)

    # STEP 3: Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_numeric, alpha=0.8, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette="Set2", s=50)
    plt.title("SVM Decision Boundary in PCA Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

except Exception as main_error:
    debug_print("Error during entire process", str(main_error))












