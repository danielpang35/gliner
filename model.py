from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

labels = ["match"]

prompt="""You are trying to find what amenities, features, and accomodations makes a glamping trip special to visitors and guests. 
Find all positive aspects of the glamping experience described by the review below, including special locations or services provided which are described by the review. """

reviews = []
reviews.append("""This was our first time staying in a yurt and we loved it. Spoiler alert: if you like to be surprised and discover everything yourself, stop reading. 
               If you like to be forewarned and well-prepared, continue. We stayed in one of the larger yurts- #2- and found it to be very comfortable. We brought our coffee-maker and used it. 
               There were also plenty of other outlets for charging cellphones or for a night light. There is a ceiling fan with adjustable speeds.
                We were there during heavy rain. While the yurt does not leak, rain can blow in from the windows. We did not want to go outside and lower the flap that covers the open screened in window, 
               but I did have to protect our bedding from rain that blew in. Fortunately, I had something to use.
                It is a bit of a walk to the bathroom and you'll need a flashlight. There is an outdoor light for the yurt and the bathrooms are lighted, but it is still a dark route down the road.
                I'm already thinking about my next yurt stay- perhaps here ( the area is rich with culture and the Appalachian Trail) or maybe another state park. I definitely want to do it again.""")

reviews.append("""Our family of 4 adults, really enjoyed our time here, the cabin was comfortable and very clean. The hosts were incredibly kind and welcoming. 
               We absolutely loved interacting with all the petting zoo animals, especially the donkey and alpacas. 
               We spent much of our time sitting on the front porch relaxing with our 2 dogs, reading books and also in the lodge playing cards and games. 
               Very enjoyable! Would definitely go back!""")

reviews.append("""My wife and I needed a short getaway, and this was the perfect stay for us. 
               We took our pups and board games, and whiled away the weekend alternately enjoying the outdoor activities (a GREAT petting zoo with loved and loving animals, not one of those sad ones, and a small pond for swimming) and playing games or relaxing in our suite. 
               We had planned to drive to a nearby lake to do some boating, but ended being relaxed enough here that we didn't bother.""")

reviews.append("""Unique awesome stay. Hostess was quick to respond, loved the fact that there was a book of how this place came to be. 
               Cute interior, nice deck, hot tub and firepit, bed was comfortable, nice old sliding bathroon door, good water pressure. 
               Drive is a challenge but ok. Would definitely stay again. Great stay. Great memories.""")
for review in reviews:
    input = prompt+review
    matches = model.predict_entities(input, labels)
    for match in matches:
        print(match["text"], "=>", match["score"])
