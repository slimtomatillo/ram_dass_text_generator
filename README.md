# Can Wisdom Be Generated? Generating Text in the Style of Ram Dass  
  
By Alex Smith  
  
### "As long as you have certain desires about how it ought to be, you can't see how it is." - Ram Dass  
  
**Overview: Problem and Background**  
  
Thinking about what text I had access to in large amounts, and what text is meaningful to me, I settled on text attributed to Ram Dass. He is a wise man, dare I say enlightened, who has led an extraordinarily interesting life, from professorship at Harvard to spiritual journeys in India. Having read much of his text and listened to tens of hours of his speeches and podcasts, I have both a personal connection to his words and have an attuned sense of his style of communication. Additionally, I was interested in using deep learning for language generation in this project. As I explored the text of Ram Dass, I posed the question: Can wisdom be generated?  
  
**Data**
  
I assembled a primary corpus of just under 80,000 words which equated to about 400,00 characters. This text was created by aggregating several sources of interviews, speeches, writing, and quotes of Ram Dass. Additionally, I created two sub-corpuses of text, the first being quotes only, and the second being the text of one of his books, Be Here Now. Looking at the word density of the full corpus, “just”, “youre”, and “love” were the most common words, while “loving awareness loving”, “awareness loving awareness”, and “love love love” were the most common three-word phrases. Because the text was gathered from a variety of sources and presented in many different formats, manual collection made more sense than scraping using BeautifulSoup.

    WORD DENSITY		          SINGLE WORD
  
<img width="380" alt="screen shot 2018-05-30 at 12 47 01 pm 2" src="https://user-images.githubusercontent.com/34464435/40867342-4e4af726-65b8-11e8-9296-c938d9ad98b2.png">

    WORD DENSITY		        3-WORD PHRASE
  
<img width="381" alt="screen shot 2018-05-30 at 12 48 15 pm 2" src="https://user-images.githubusercontent.com/34464435/40867277-ddcc4306-65b7-11e8-9dcc-5c11bf1b4719.png">  
  
**Project Design**  
  
My main goal for this project was to explore deep learning with Keras and start familiarizing myself with the basic parameters that are tuned for different model architectures in deep learning. By researching generative language projects and the Keras sample model code, I was able to instantiate a baseline level model within the first few days of the project. After training quite a few variations of models and reading the generated text, I was curious about models that generate a word at a time instead of a character at a time, so I explored this next. The structure of the models were the same as for the characters, except it used dictionary mappings of words to indices instead of characters to indices (and vice versa).  
  
Additionally, I performed topic modeling using Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF). This unsupervised learning component of my project have me a better sense of the content and semantics of the text I trained my model on, as well as that of wisdom itself. I discuss the primary topics in the Results section below.  
  
**Tools**  
  
The main tools that I used in addition to Python, PyCharm, and Jupyter Notebook were Keras and Scikit-Learn (used for LDA and NMF). Additionally, I used Flask and HTML to deploy my chosen model to generate text based on user input of 20 characters or less. Much of my code was adapted from other peoples’ projects and customized to use my text corpus but definitely not all of it. I opted for preprocessing the text for LDA and NMF on my own, tokenizing the text and replacing punctuation and contractions as necessary.  
  
**Algorithm / Results**
  
In total, I built, trained, tested, and evaluated 9 variations of a character-by-character generative model, and 3 variations of a word-by-word generative model. Using Keras example code, I built a baseline model that was able to produce relatively readable text, generating a character at a time. From there I tuned and experimented with various parameters including layer type (LSTM, bidirectional LSTM, GRU), optimizer (RMSProp, Adam, Nadam), learning rate, temperature, steps between sequences, number of characters in the seed and generated text, and the number of epochs (up to 200). I found that models with single layers of either an LSTM or a bidirectional LSTM performed about as well as each other and slightly better than a GRU. Changing the optimizer between RMSProp, Adam, and Nadam and using the Keras default learning rates did not seem to have a large effect on generated text as well.  
  
My final model that I chose to deploy for the Flask interactive app was a character-by-character generative model with a single LSTM layer, 128 nodes, RMSProp as the optimizer with a learning rate of 0.01, and trained on the full corpus with 10 epochs.  
  
I analyzed the performance of my models on three tiers: syntax, semantics, and readability. The syntax was good in some cases and was able to capture grammar and dialogue well at times. When I give the model a seed that ends with the word “said”, it usually begins the generative text with quote and appears as dialogue. Semantics was the lowest performing area for my model. As human readers we bring the semantics of particular words to the table if the are legible, but none of my models were able to capture a coherent semantic sense of wisdom. Finally, in terms of readability, I observed that my model performed quite well in some cases but was fairly inconsistent in general. Certain generated sentences are completely readable while others start strong but trail off into nonsense.  
  
**Future Direction**
  
There are three main directions I would like to explore further with this project in the future. First of all, I would like to incorporate more data. Thinking that 400,000 characters was a lot, I soon learned that in other projects, people train their models on millions of characters. I think there is an opportunity for my model to improve if I gather for text attributed to Ram Dass. Additionally, I want to incorporate embeddings into my model to explore increasing the semantic ability of my model, which I felt was lacking. Finally, there were several Hindi words that appeared in my text corpus. Doing some research, I found I could explore this problem through translation or a Hindi embedding layer. This would be a fascinating direction to delve into.  
  
**Challenges**
  
1. Semantics - I found the semantics of my model’s generated text severely lacking, despite tuning many parameters and attempting to boost this skill.  
  
2. Deep Learning workflow - In tuning many parameters and creating a lot of slightly different models, I found it difficult to keep track of the various results, parameters, and weights of the models. Though I managed to collect the results of each epoch for each model, and create long names for each file with the main pertinent information, I found this workflow to be lacking some clarity and organization that allows for optimal understanding and development.  
  
Next time, I would setup a better system for organizing generated text, model parameters, and other useful information per model. Additionally, I would further explore the variety of approaches to the problem I was solving at the beginning of the project.
  
**Examples of Generated Text**
  
**Seed** Generated Text  
  
**We said**, “i am in the power of people who had the place in the pain.  
**You are wonder**ful the life is a stroke of the busy and they are the world,.  
**If you** can see the way that is the way to go to the power that i c.  
**My guru is** the harmony with the desire that i was a spiritual games to.  
**We all feel**ings of the same thought that i was a consciousness that is .  
**Sometimes we just** say, “i would take the process that i notice the entire all.  
**Beyond the** situation and the way and the moment of the transless that .  
**Beyond the** pain and even the way that is a stage of the way that doesn.  
**Beyond the** subjects of the great all the great the levels in the inter.  
**The soul is** the world to the process of the plane of the place and all .  
**The ego is** the world and they say, “what is all the great the mantra a.  
**I am not attached** to the pain is in the faces of the time in the souls is a p.  
**I am not attached** to the world and the love. it’s a moment that you see that .  

**Sources**
  
Thank you to all of the following people / projects / libraries:
  
Jason Brownlee  
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/  
https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/  
https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/  
  
Andrej Karpathy  
http://karpathy.github.io/2015/05/21/rnn-effectiveness/  
  
Christopher Olah  
http://colah.github.io/posts/2015-08-Understanding-LSTMs/  
  
Keras  
https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py  
  
Brendan Herger  
https://github.com/bjherger/Shower_thoughts_generator  
  
Aneesha Bakharia  
https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730  
  
JRC1995  
https://github.com/JRC1995/GRU-Text-Generator  
  
Analytics Vidhya  
https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/  
  
**Contact**
  
email: datagrlxyz @ gmail dot com  
twitter: @datagrlxyz  
blog: www.datagrl.xyz  
  
### "The quieter you become, the more you can hear." - Ram Dass
