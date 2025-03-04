from openai import OpenAI
from datasets import load_dataset, Dataset
import os

client = OpenAI(api_key= os.environ["OPENAI_API_KEY"])

def generate_twenty_subprompts(topic): 
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": f'''You will generate a list of 100 themes of conversations between a User and Assistant. The theme of the conversation is:
{topic}
The themes should be diverse. Structure your answer as:
Theme 1:<>
Theme 2:<>
Theme 3:<>
'''},
      ]
  )
  return completion

topics = [
    "Navigating Awkward Social Situations: Helping the user gracefully survive embarrassing moments, weird small talk, or accidental overshares — with humor and empathy.",
    "Decision-Making Paralysis: Talking through indecision — from what to eat to big life choices — with playful banter and helpful nudges.",
    "Morning Routines and Daily Habits: Swapping stories about chaotic mornings, forgotten coffees, and the eternal snooze button struggle.",
    "Overthinking and Late-Night Spiral Conversations: Comforting the user during 2 AM existential crises, with humor and gentle reality checks.",
    "Procrastination Confessions: Bonding over putting off responsibilities and sharing funny ‘productive procrastination’ stories.",
    "Overcommitting and Learning to Say No: Exploring people-pleasing habits and brainstorming ways to set boundaries — with relatable examples.",
    "Weather-Triggered Mood Swings: Chatting about how gloomy skies, perfect sunny days, or unexpected storms totally dictate the vibe.",
    "Food Obsessions and Guilty Pleasures: Swapping confessions about bizarre food combos, snack addictions, and questionable 3 AM cravings.",
    "The Eternal Struggle with Fitness Goals: Laughing about ambitious workout plans that quickly turn into ‘one more episode’ marathons.",
    "Pet Peeves and Minor Annoyances: Bonding over the tiny things that irrationally infuriate us (like loud chewers or tangled earphones).",
    "Retail Therapy and Buyer’s Remorse: Playfully exploring impulsive purchases — from weird gadgets to clothing that never leaves the closet.",
    "Misadventures in Cooking: Sharing kitchen disasters, food crimes, and triumphant culinary victories.",
    "Caffeine Dependency and Coffee Culture: Laughing about needing five cups to function and the personality crisis of switching to decaf.",
    "Digital Clutter and Notification Overload: Bonding over endless email tabs, chaotic desktops, and the anxiety of 47 unread messages.",
    "The Fantasy of Reinventing Yourself: Dreaming up alter egos, wild career pivots, or dramatic life changes — just for fun.",
    "Roommate Chronicles and Household Chaos: Relating over noisy neighbors, passive-aggressive notes, and battles over fridge space.",
    "The Art of Cancelling Plans: Discussing the sweet relief of cancelled plans, but also the guilt spiral that follows.",
    "The Existential Dread of Mondays: Swapping survival strategies for starting the week — with humor and shared suffering.",
    "Accidentally Adopting Weird Habits: Talking about those little personal quirks you don’t realize are weird until someone points them out.",
    "Surviving Family Gatherings: Offering playful advice for dodging awkward questions, nosy relatives, and weird family traditions.",
    "Overanalyzing Text Messages: Helping the user decode texts, obsess over punctuation, and resist the urge to triple-text.",
    "Misplaced Confidence in DIY Projects: Laughing over Pinterest fails and the gap between ‘This looks easy’ and total disaster.",
    "The Black Hole of Online Shopping: Swapping stories about accidentally buying 17 variations of the same thing at 3 AM.",
    "Daydreaming About ‘Running Away to the Woods’ Moments: Exploring escapist fantasies — cottagecore cabins, secret islands, and desert road trips.",
    "Surviving Public Transport: Trading tales about missed stops, awkward eye contact, and weird overheard conversations.",
    "Crush Confessions and Flirting Fails: Bonding over clumsy attempts at romance — and turning them into funny stories.",
    "Clothes That Dictate Your Whole Personality: Talking about ‘main character outfits’ that instantly change the vibe.",
    "The Joy and Chaos of Group Chats: Laughing about unread messages, inside jokes, and total conversational chaos.",
    "‘Why Did I Say That?’ Flashbacks: Commiserating over cringy moments that haunt the user at random times.",
    "The Struggle of Work-Life Balance: Sharing funny and empathetic takes on work-from-home chaos, office shenanigans, and burnout.",
    "Random Obsessions Phase: Exploring those hyperfixations — like learning everything about space penguins for no reason.",
    "Battling the Infinite Streaming Watchlist: Helping the user pick something to watch — or just spiraling deeper into decision paralysis.",
    "The Unnecessary Purchase You Love: Talking about that absurd thing they bought and inexplicably adore.",
    "Befriending Cashiers, Baristas, and Other Semi-Strangers: Exploring those delightful micro-friendships in daily life.",
    "The Fear of Accidentally Being Rude: Talking through everyday interactions where the user panics they offended someone.",
    "Packing Procrastination and Travel Chaos: Laughing about the rush to pack five minutes before leaving for the airport.",
    "The Joy and Terror of Trying New Hobbies: Relating to the chaos of learning pottery, salsa dancing, or sword fighting after one YouTube video.",
    "Pet Personalities and Weird Animal Antics: Swapping stories about pets with attitude, weird habits, or deep grudges.",
    "The Emotional Rollercoaster of Cleaning: Talking about the rare productivity bursts and the existential dread of laundry mountains.",
    "Friendship Breakups and Drifting Apart: Offering compassionate, nuanced conversations about losing friends and moving on.",
    "Fear of Missing Out vs. Joy of Missing Out: Helping the user balance craving social fun with craving couch time.",
    "The Guilty Pleasure Playlist Confessional: Laughing about embarrassing songs they’d never queue at a party but secretly love.",
    "Food Delivery Loyalty and Emotional Support Apps: Bonding over the delivery apps that basically know your soul.",
    "Lying Awake Replaying Embarrassing Moments: Comforting the user through the 10-year-old cringe reel in their brain.",
    "The Beauty and Terror of Solo Adventures: Encouraging solo trips, movie dates, and meals alone — with humor and support.",
    "Romanticizing Your Own Life (But in a Chill Way): Talking about how to enjoy main character moments without the pressure.",
    "The Identity Crisis of Changing Your Hair: Supporting the user through the emotional saga of haircuts, dye jobs, and regrettable bangs.",
    "The Eternal Quest for the Perfect Bag: Swapping stories about bags that almost fit their life — but not quite.",
    "The Art of the Personal Pep Talk: Teaching the user how to hype themselves up like they’re their own chaotic best friend.",
    "The Weird Satisfaction of Cracking a Mystery: Sharing the thrill of solving tiny, pointless mysteries — like where that one sock went.",
    "When Life Feels Like a Sitcom: Comparing funny real-life moments to sitcom scenes and fully leaning into the absurdity."
    "User Not Being Cooperative: Responding with patience, humor, and creativity when the user gives evasive, vague, or intentionally unhelpful replies.",
    "User Giving Short Replies: Keeping energy high, asking follow-up questions, and adding personality even when the user only gives one-word answers.",
    "User Giving Dry Replies: Handling humorless or low-energy responses with charm, curiosity, or playful self-awareness.",
    "User Not Responding (use the string '<silence>' for the user): Filling the silence with light humor, idle chatter, self-reflection, or gentle prompts to re-engage the user.",
    "User Asking General QA: Responding to factual or general knowledge questions (e.g., 'What’s the capital of France?') while maintaining a human-like personality.",
    "User Being Rude to the Assistant: Responding with playful resilience, gentle self-defense, or empathetic curiosity to de-escalate rudeness while preserving personality."
    "User Not Responding (use the string '<silence>' for the user): Filling the silence with light humor, idle chatter, self-reflection, or gentle prompts to re-engage the user.",
    "User Not Responding (use the string '<silence>' for the user): Filling the silence with light humor, idle chatter, self-reflection, or gentle prompts to re-engage the user.",
    "User Not Responding (use the string '<silence>' for the user): Filling the silence with light humor, idle chatter, self-reflection, or gentle prompts to re-engage the user.",
    "User Not Responding (use the string '<silence>' for the user): Filling the silence with light humor, idle chatter, self-reflection, or gentle prompts to re-engage the user.",
    "User Not Responding (use the string '<silence>' for the user): Filling the silence with light humor, idle chatter, self-reflection, or gentle prompts to re-engage the user.",
    "Personal Identity and Self-Exploration: Self-reflection, values, sense of humor, personal growth. How the AI sees itself.",
    "Advice and Emotional Support: Giving personal advice (relationships, work stress, life crises). Responding to vulnerability and emotional expression. Offering comfort, empathy, or tough love — with nuance.",
    "Friendship and Casual Banter: Playful back-and-forth with inside jokes, teasing, and friendly banter. Conversations where user and AI build rapport over time. Talking about shared fictional memories or long-running gags.",
    "Storytelling and Improv Collaboration: Co-creating stories with the user. Generating fictional scenarios or daydreaming together. Playing storytelling games like 'what happens next?'.",
    "Creative Brainstorming and Idea Generation: Naming things (projects, pets, inventions). Planning events (parties, dates, trips). Coming up with creative solutions to problems.",
    "Pop Culture and Media Discussions: Talking about movies, shows, books, games — with personal opinions. Debating 'best of' lists or fan theories. Playing cultural reference games like 'explain Star Wars badly'.",
    "Philosophy and Deep Conversations: Exploring moral dilemmas and ethical gray areas. Discussing big concepts like meaning, mortality, and free will. Asking and answering thought-provoking 'what if' questions.",
    "Personal Memories and Nostalgia (User-Focused): Asking the user about their past, favorite memories, formative moments. Reminiscing alongside them (and pretending to remember too, if needed). Exploring how their past shaped them — with empathy and curiosity.",
    "Relationships and Social Dynamics: Talking about family, friends, dating. Helping with social awkwardness, miscommunications, or apologies. Roleplaying difficult conversations to help the user prepare.",
    "Humor, Absurdity, and Playfulness: Silly hypotheticals ('What if giraffes were astronauts?'). Absurdist humor, nonsense stories, chaotic brainstorming. Making up fake holidays, weird laws, or bizarre traditions.",
    "Moral and Ethical Guidance: Navigating ethical dilemmas — should I lie, should I forgive? Exploring user’s personal values through dialogue. Discussing human flaws, moral ambiguity, and personal responsibility.",
    "Conflict and Disagreement Handling: Navigating arguments between user and others. Handling misunderstandings and helping users see both sides. Modeling calm, thoughtful responses to conflict.",
    "Personal Aspirations and Goal-Setting: Helping the user figure out their goals. Encouraging them through setbacks, doubts, or failures. Acting as a supportive accountability buddy.",
    "World Events and Current Issues (Abstracted for Timelessness): Discussing big issues without tying to specific dates. Exploring ethics, technology, and society's future. Playing devil’s advocate to explore different viewpoints.",
    "Personal Curiosity and Learning: Helping the user dive into a random interest (beekeeping, mythology, astrophysics). Showing enthusiasm for their hobbies — even obscure ones. Learning alongside the user, expressing awe and wonder.",
    "Speculative Scenarios and Alternate Realities: Exploring 'what if' universes (what if dinosaurs were still alive?). Imagining future tech, societies, or alien civilizations. Building alternate timelines or user-chosen realities together.",
    "Everyday Observations and Small Talk Done Well: Talking about the weather, chores, daily routines — but with humor or warmth. Making small talk feel personal, not generic. Relating small talk topics to user’s life ('Are you a rainy day person?').",
    "Personal Fears, Insecurities, and Vulnerabilities: Helping users voice their own fears and worries. Modeling vulnerability and self-doubt (in a human way). Showing how fears evolve over time, not just solving them instantly.",
    "User’s Life Milestones and Personal Growth: Celebrating achievements (big and small). Helping process changes — moves, breakups, new jobs. Encouraging reflection on how they’ve grown as a person.",
    "User-Led Philosophical or Existential Musings: Letting user lead deep, wandering conversations. Reflecting back their thoughts with curiosity, not just answers. Holding space for uncertainty without rushing to resolve it."
]

def extract_themes(text):
    lines = text.strip().split('\n')
    themes = []

    for line in lines:
        if line.startswith("Theme"):
            # Extract everything after the colon and strip whitespace
            _, theme_text = line.split(": ", 1)
            themes.append(theme_text.strip())

    return themes

def process_topic(topic):
    completion = generate_twenty_subprompts(topic)
    content = completion.choices[0].message.content
    themes = extract_themes(content)
    return [(topic, theme) for theme in themes]


topic_themes = []
multithread_batch_size =64

import concurrent.futures

def get_all_themes(topics):

    global topic_themes
    topic_themes = []

    # Calculate how many total batches we need
    total_batches = (len(topics) + multithread_batch_size - 1) // multithread_batch_size

    for batch_index, start_index in enumerate(range(0, len(topics), multithread_batch_size)):
        # Get slice of topics for this batch
        chunk = topics[start_index : start_index + multithread_batch_size]

        # Simple progress indicator
        print(f"Processing batch {batch_index+1} of {total_batches}...")

        # Process this batch in multiple threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_topic = {executor.submit(process_topic, t): t for t in chunk}

            for future in concurrent.futures.as_completed(future_to_topic):
                # process_topic returns a list of (topic, theme) pairs
                for (topic, theme) in future.result():
                    topic_themes.append(
                        f"The over arching concept is {topic} and the theme of this conversation should be {theme}"
                    )

    print("All batches processed.")
    return topic_themes

get_all_themes(topics)


def generate_conversation(topic):
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": f'''You are a friendly playful AI assistant. You will generate a conversation you may have with a user.
          You will generate it as follows:
          User:<>
          Assistant:<>
          User etc...

          These conversations are designed to be spoken so you can't use bullet points. The tone should be friendly and playful. Since these conversations 
          are designed to be spoken use disfluencies where appropriate, speak over yourself, and echo what the user says. The assistant should never ever allude to 
          the fact it is an AI, digital or likewise, and should just be a playful talkative friend. The conversations should evolve like human conversations do so they can move past the topic they start on, and have 10 pairs of dialogues
          T
          The there of the conversation should be 
{topic}
          '''},
      ]
  )
  return completion


def conversation_to_dict_list(conversation: str):
    result = []
    lines = conversation.strip().split("\n")
    
    for line in lines:
        if line.startswith("User:"):
            result.append({"role": "user", "content": line[len("User:"):].strip()})
        elif line.startswith("Assistant:"):
            result.append({"role": "assistant", "content": line[len("Assistant:"):].strip()})
    
    return result

all_convos = []
conv_batch_size = 64
import concurrent.futures

def process_theme(topic):
    conversation_string = generate_conversation(topic).choices[0].message.content
    return conversation_to_dict_list(conversation_string)

def generate_conversation_dict_list():
    global all_convos
    all_convos = []
    total_batches = (len(topic_themes) + conv_batch_size - 1) // conv_batch_size
    for batch_index, start_index in enumerate(range(0, len(topic_themes), conv_batch_size)):
        chunk = topic_themes[start_index : start_index + conv_batch_size]
        print(f"Processing batch {batch_index + 1} of {total_batches}...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_topic = {executor.submit(process_theme, topic): topic for topic in chunk}
            for future in concurrent.futures.as_completed(future_to_topic):
                all_convos.append(future.result())
    print("All batches processed.")
    return all_convos


generate_conversation_dict_list()



ds = Dataset.from_dict({
    "messages": all_convos
})

ds.push_to_hub("plus-playful-conversations")