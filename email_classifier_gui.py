"""
Aplikacja GUI do klasyfikacji emaili
Prosta graficzna wersja klasyfikatora
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

def softmax(x):
    """Konwertuje decision scores na prawdopodobieństwa"""
    exp_x = np.exp(x - np.max(x))  # stabilność numeryczna
    return exp_x / exp_x.sum()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    import spacy
    nlp_pl = spacy.load('pl_core_news_sm')
except:
    import sys
    print("⚠️ Instalowanie modelu spaCy...")
    os.system('python -m spacy download pl_core_news_sm')
    import spacy
    nlp_pl = spacy.load('pl_core_news_sm')

# Pobieranie zasobów NLTK (dla angielskiego)
try:
    stop_words_en = set(stopwords.words('english'))
except:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words_en = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Polskie stop words
stop_words_pl = {
    'i', 'w', 'z', 'na', 'do', 'o', 'się', 'nie', 'to', 'jest', 'że', 'a',
    'co', 'jak', 'za', 'od', 'po', 'ale', 'czy', 'tak', 'są', 'my', 'wy',
    'on', 'ona', 'ono', 'oni', 'one', 'ja', 'ty', 'mu', 'go', 'jej', 'ich',
    'mi', 'mnie', 'cię', 'ci', 'was', 'nam', 'wam', 'im', 'niego', 'niej',
    'tego', 'tej', 'tym', 'tych', 'tym', 'temu', 'tą', 'te', 'ta', 'tę',
    'ze', 'przez', 'dla', 'przy', 'pod', 'nad', 'przed', 'między', 'bez',
    'będzie', 'został', 'została', 'zostało', 'może', 'mogą', 'można',
    'kiedy', 'gdzie', 'gdy', 'lub', 'oraz', 'albo', 'ani', 'więc', 'jednak',
    'także', 'również', 'właśnie', 'bowiem', 'ponieważ', 'gdyż', 'aby',
    'żeby', 'sobie', 'siebie', 'sobą', 'jako', 'który', 'która', 'które',
    'których', 'którym', 'którą', 'których', 'czym', 'kto', 'kim', 'komu',
    'być', 'mieć', 'móc', 'chcieć', 'zostać', 'ten', 'bardzo', 'też',
    'tylko', 'jeszcze', 'już', 'teraz', 'tutaj', 'tam', 'tu', 'wszystko',
    'wszystkie', 'wszystkich', 'każdy', 'każda', 'każde', 'każdego',
    'jakiś', 'jakaś', 'jakieś', 'jakichś', 'coś', 'ktoś', 'nic', 'nikt',
    'zawsze', 'nigdy', 'często', 'rzadko', 'czasem', 'potem', 'teraz',
    'wcześniej', 'później', 'dzisiaj', 'jutro', 'wczoraj', 'roku', 'lat'
}

URL_PATH_PATTERN = r'(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/\S*)?\b)'

def clean_raw_noise(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(URL_PATH_PATTERN, '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'(/[a-zA-Z0-9._-]+)+|([a-zA-Z]:\\[a-zA-Z0-9._\\]+)', '', text)
    return text

def preprocess_text_pl(text):
    text = clean_raw_noise(text)
    text = re.sub(r'[^a-ząćęłńóśźż\s]', ' ', text)
    text = ' '.join(text.split())
    
    doc = nlp_pl(text)
    tokens = [token.lemma_ for token in doc
              if token.lemma_ not in stop_words_pl
              and len(token.lemma_) > 2
              and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

def preprocess_text_en(text):
    text = clean_raw_noise(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words_en and len(word) > 2]
    return ' '.join(tokens)


class EmailClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📧 Klasyfikator Emaili")
        self.root.geometry("800x750")
        self.root.resizable(True, True)
        
        self.models = {}  # {'polish': {...}, 'english': {...}}
        self.current_language = 'polish'
        
        # Kolory dla kategorii
        self.category_colors = {
            'forum': '#FF6B6B',
            'promotions': '#4ECDC4',
            'social_media': '#45B7D1',
            'spam': '#FFA07A',
            'updates': '#98D8C8',
            'verify_code': '#DDA0DD'
        }
        
        self.setup_ui()
        self.load_model_auto()
        
    def setup_ui(self):
        """Tworzy interfejs użytkownika"""
        # Nagłówek
        header_frame = tk.Frame(self.root, bg='#2C3E50', height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="📧 Klasyfikator Emaili",
            font=('Arial', 20, 'bold'),
            bg='#2C3E50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Status model
        self.status_frame = tk.Frame(self.root, bg='#ECF0F1', height=40)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="⏳ Ładowanie modelu...",
            font=('Arial', 10),
            bg='#ECF0F1',
            fg='#34495E'
        )
        self.status_label.pack(pady=10)
        
        # Główny kontener
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Pole tekstowe do wprowadzania
        input_label = tk.Label(
            main_frame,
            text="Wprowadź tekst emaila:",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        input_label.pack(anchor='w', padx=5, pady=(10, 5))
        
        self.text_input = scrolledtext.ScrolledText(
            main_frame,
            height=10,
            font=('Arial', 11),
            wrap=tk.WORD,
            borderwidth=2,
            relief=tk.GROOVE
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Przyciski
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Przyciski wyboru języka
        lang_frame = tk.Frame(button_frame, bg='white')
        lang_frame.pack(side=tk.LEFT, padx=5)
        
        self.lang_pl_btn = tk.Button(
            lang_frame,
            text="🇵🇱 Polski",
            font=('Arial', 10, 'bold'),
            bg='#3498DB',
            fg='white',
            activebackground='#2980B9',
            command=lambda: self.set_language('polish'),
            cursor='hand2',
            padx=15,
            pady=8
        )
        self.lang_pl_btn.pack(side=tk.LEFT, padx=2)
        
        self.lang_en_btn = tk.Button(
            lang_frame,
            text="🇬🇧 English",
            font=('Arial', 10),
            bg='#95A5A6',
            fg='white',
            activebackground='#7F8C8D',
            command=lambda: self.set_language('english'),
            cursor='hand2',
            padx=15,
            pady=8
        )
        self.lang_en_btn.pack(side=tk.LEFT, padx=2)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="🔍 Klasyfikuj",
            font=('Arial', 12, 'bold'),
            bg='#3498DB',
            fg='white',
            activebackground='#2980B9',
            command=self.classify_email,
            cursor='hand2',
            padx=20,
            pady=10
        )
        self.classify_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Wyczyść",
            font=('Arial', 12),
            bg='#95A5A6',
            fg='white',
            activebackground='#7F8C8D',
            command=self.clear_all,
            cursor='hand2',
            padx=20,
            pady=10
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        example_btn = tk.Button(
            button_frame,
            text="💡 Przykład",
            font=('Arial', 12),
            bg='#9B59B6',
            fg='white',
            activebackground='#8E44AD',
            command=self.load_example,
            cursor='hand2',
            padx=20,
            pady=10
        )
        example_btn.pack(side=tk.LEFT, padx=5)
        
        # Wyniki
        result_label = tk.Label(
            main_frame,
            text="Wyniki klasyfikacji:",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        result_label.pack(anchor='w', padx=5, pady=(10, 5))
        
        self.result_frame = tk.Frame(main_frame, bg='white')
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_model_auto(self):
        """Automatyczne ładowanie modelu"""
        model_file = 'email_classifier_model.pkl'
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    self.models = pickle.load(f)
                
                self.status_label.config(
                    text="✅ Modele załadowane pomyślnie! (🇵🇱 Polski + 🇬🇧 English)",
                    fg='#27AE60'
                )
                self.classify_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.status_label.config(
                    text=f"❌ Błąd ładowania modelu: {str(e)}",
                    fg='#E74C3C'
                )
                self.classify_btn.config(state=tk.DISABLED)
        else:
            self.status_label.config(
                text="❌ Nie znaleziono modelu! Uruchom komórkę 19 w notatniku project.ipynb",
                fg='#E74C3C'
            )
            self.classify_btn.config(state=tk.DISABLED)
    
    def set_language(self, language):
        """Zmienia język klasyfikacji"""
        self.current_language = language
        
        # Aktualizacja wyglądu przycisków
        if language == 'polish':
            self.lang_pl_btn.config(bg='#3498DB', font=('Arial', 10, 'bold'))
            self.lang_en_btn.config(bg='#95A5A6', font=('Arial', 10))
        else:
            self.lang_en_btn.config(bg='#3498DB', font=('Arial', 10, 'bold'))
            self.lang_pl_btn.config(bg='#95A5A6', font=('Arial', 10))
    
    def classify_email(self):
        """Klasyfikuje wprowadzony email"""
        text = self.text_input.get('1.0', tk.END).strip()
        
        if not text:
            messagebox.showwarning("Uwaga", "Wprowadź tekst emaila!")
            return
        
        if not self.models:
            messagebox.showerror("Błąd", "Modele nie zostały załadowane!")
            return
        
        # Czyszczenie poprzednich wyników
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        try:
            model_data = self.models[self.current_language]
            
            # Preprocessing (wybierz odpowiednią funkcję)
            if self.current_language == 'polish':
                clean_text = preprocess_text_pl(text)
            else:
                clean_text = preprocess_text_en(text)
            
            # Wektoryzacja
            text_vector = model_data['vectorizer'].transform([clean_text])
            
            # Predykcja
            prediction = model_data['model'].predict(text_vector)[0]
            category = model_data['label_encoder'].inverse_transform([prediction])[0]
            
            # Prawdopodobieństwa dla wszystkich kategorii
            decision_scores = model_data['model'].decision_function(text_vector)[0]
            probabilities = softmax(decision_scores) * 100  # Konwersja na procenty
            
            # Wyświetlanie głównego wyniku
            main_result = tk.Frame(self.result_frame, bg='white')
            main_result.pack(fill=tk.X, pady=10)
            
            lang_emoji = "🇵🇱" if self.current_language == 'polish' else "🇬🇧"
            result_text = tk.Label(
                main_result,
                text=f"{lang_emoji} {category.upper()}",
                font=('Arial', 16, 'bold'),
                bg=self.category_colors.get(category, '#95A5A6'),
                fg='white',
                padx=20,
                pady=10,
                borderwidth=2,
                relief=tk.RAISED
            )
            result_text.pack()
            
            # Szczegółowe wyniki
            details_label = tk.Label(
                self.result_frame,
                text="Prawdopodobieństwa:",
                font=('Arial', 11, 'bold'),
                bg='white'
            )
            details_label.pack(anchor='w', pady=(10, 5))
            
            # Sortowanie wyników
            results = list(zip(model_data['label_encoder'].classes_, probabilities))
            results.sort(key=lambda x: x[1], reverse=True)
            
            for cat, prob in results:
                self.create_score_bar(cat, prob, cat == category)
                
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas klasyfikacji:\n{str(e)}")
    
    def create_score_bar(self, category, probability, is_predicted):
        """Tworzy pasek prawdopodobieństwa dla kategorii"""
        frame = tk.Frame(self.result_frame, bg='white')
        frame.pack(fill=tk.X, pady=2)
        
        # Nazwa kategorii
        name_label = tk.Label(
            frame,
            text=category,
            font=('Arial', 10, 'bold' if is_predicted else 'normal'),
            bg='white',
            width=15,
            anchor='w'
        )
        name_label.pack(side=tk.LEFT, padx=5)
        
        # Pasek postępu
        bar_frame = tk.Frame(frame, bg='#ECF0F1', height=25, width=400)
        bar_frame.pack(side=tk.LEFT, padx=5)
        bar_frame.pack_propagate(False)
        
        # Wypełnienie paska (0-100%)
        bar_width = int(400 * probability / 100)
        
        bar = tk.Frame(
            bar_frame,
            bg=self.category_colors.get(category, '#95A5A6'),
            width=bar_width,
            height=25
        )
        bar.pack(side=tk.LEFT)
        
        # Wartość w procentach
        score_label = tk.Label(
            frame,
            text=f"{probability:.1f}%",
            font=('Arial', 10, 'bold' if is_predicted else 'normal'),
            bg='white',
            width=8
        )
        score_label.pack(side=tk.LEFT, padx=5)
        
        # Znacznik
        if is_predicted:
            marker = tk.Label(
                frame,
                text="👈",
                font=('Arial', 12),
                bg='white'
            )
            marker.pack(side=tk.LEFT)
    
    def clear_all(self):
        """Czyści wszystkie pola"""
        self.text_input.delete('1.0', tk.END)
        for widget in self.result_frame.winfo_children():
            widget.destroy()
    
    def load_example(self):
        """Ładuje przykładowy email"""
        examples_pl = [
            "Twój kod weryfikacyjny to 847291. Kod wygasa za 10 minut.",
            "Gratulacje! Wygrałeś 1 000 000 zł! Kliknij tutaj, aby odebrać nagrodę!",
            "Sara oznaczyła Cię na nowym zdjęciu. Zobacz, co robią Twoi znajomi!",
            "Błyskawiczna wyprzedaż! 50% zniżki na wszystko! Użyj kodu SAVE50.",
            "Nowa odpowiedź w Twoim wątku o bibliotekach Pythona do uczenia maszynowego.",
            "Dziękujemy za subskrypcję! Oto najnowsze artykuły z naszego bloga."
        ]
        
        examples_en = [
            "Your verification code is 847291. This code expires in 10 minutes.",
            "Congratulations! You won $1,000,000! Click here to claim your prize!",
            "Sarah tagged you in a new photo. See what your friends are up to!",
            "Flash Sale! 50% off all items! Use code SAVE50 at checkout.",
            "New reply to your thread about Python libraries for machine learning.",
            "Thank you for subscribing! Here are the latest articles from our blog."
        ]
        
        import random
        if self.current_language == 'polish':
            example = random.choice(examples_pl)
        else:
            example = random.choice(examples_en)
        
        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', example)


def main():
    root = tk.Tk()
    app = EmailClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
