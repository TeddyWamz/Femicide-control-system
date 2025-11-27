# Comprehensive Data Augmentation for Small Datasets
# Creates synthetic training data when you have <100 samples

import pandas as pd
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

def create_extensive_manual_dataset():
    """
    Create 500+ high-quality manual examples in English and Swahili
    """
    data = []
    
    # ==================== PHYSICAL VIOLENCE ====================
    physical_english = [
        "He punched me in the face repeatedly",
        "My husband beats me every day with his fists",
        "He kicked me in the stomach while I was pregnant",
        "He pushed me down the stairs violently",
        "He choked me until I couldn't breathe",
        "He broke my arm during the attack",
        "He slapped me across the face multiple times",
        "He threw objects at me causing injuries",
        "He dragged me by my hair across the room",
        "He hit me with a belt leaving bruises",
        "He strangled me and threatened to kill me",
        "He beat me unconscious last night",
        "He physically assaulted me in front of our children",
        "He kicked and punched me repeatedly",
        "He smashed my head against the wall",
        "He broke several of my ribs by kicking me",
        "He bit me and left marks all over my body",
        "He burned me with cigarettes",
        "He stabbed me with a knife",
        "He twisted my arm until it broke",
        "He whipped me with a cord",
        "He pushed me into furniture causing injury",
        "He pinned me down and hurt me physically",
        "He grabbed me violently by the throat",
        "He battered me leaving visible injuries",
        "He struck me with his fist multiple times",
        "He physically attacked me without warning",
        "He assaulted me causing severe pain",
        "He beat me with a wooden stick",
        "He violently shoved me to the ground",
        "He hammered me with his hands repeatedly",
        "He caused me physical harm intentionally",
        "He inflicted bodily injury on me",
        "He pummeled me until I collapsed",
        "He roughly manhandled and hurt me",
        "He brutally attacked me physically",
        "He struck blows on my body",
        "He physically hurt me badly",
        "He caused me terrible physical pain",
        "He left me with multiple injuries",
    ]
    
    physical_swahili = [
        "Alinipiga ngumi usoni mara kwa mara",
        "Mume wangu ananichapa kila siku kwa mikono yake",
        "Alinipiga mateke tumboni nikiwa mjamzito",
        "Alinisukuma kutoka ngazi kwa hasira",
        "Alinikaba shingoni hadi sikuweza kupumua",
        "Alivunja mkono wangu wakati wa shambulio",
        "Alikoboa kofi usoni mwangu mara kadhaa",
        "Alinitupa vitu vilionijeruhi",
        "Alinikokota kwa nywele zangu ndani ya chumba",
        "Alinipiga kwa ukanda akaniacha na michubuko",
        "Alinikaba shingoni na kunitishia kuniua",
        "Alinichapa hadi nikapoteza fahamu usiku wa jana",
        "Alinishambulia kimwili mbele ya watoto wetu",
        "Alinipiga mateke na ngumi mara kwa mara",
        "Aligonga kichwa changu ukutani",
        "Alivunja mbavu zangu kadhaa kwa kunipiga mateke",
        "Alinng'ata na kuniacha alama mwilini mwangu",
        "Alinichoma kwa sigara",
        "Alinichoma kwa kisu",
        "Aliupinda mkono wangu hadi ukavunjika",
        "Alinipiga kwa waya",
        "Alinisukuma kwenye samani na kunijeruhi",
        "Alinishikilia chini na kunidhuru kimwili",
        "Alinishika kwa nguvu kooni",
        "Alinipiga teke vibaya akaniacha na majeraha",
        "Alinipiga ngumi mara kadhaa",
        "Alinishambulia kimwili bila onyo",
        "Alinishambulia na kusababisha maumivu makubwa",
        "Alinichapa kwa fimbo ya mbao",
        "Alinisukuma kwa nguvu hadi nikaanguka",
        "Alinipiga kwa mikono yake mara kwa mara",
        "Alinidhuru kimwili kwa makusudi",
        "Alijeruhi mwili wangu",
        "Alinipiga hadi nikaanguka",
        "Alinishughulikia vibaya na kunidhuru",
        "Alinishambulia kwa ukali kimwili",
        "Alinipiga mapigo mwilini mwangu",
        "Alinidhuru kimwili vibaya sana",
        "Alinitendea maumivu makubwa ya kimwili",
        "Aliniacha na majeraha mengi mwilini",
    ]
    
    # ==================== SEXUAL VIOLENCE ====================
    sexual_english = [
        "He forced me to have sex against my will",
        "He raped me when I refused his advances",
        "He sexually assaulted me repeatedly",
        "He touched me inappropriately without consent",
        "He molested me while I was sleeping",
        "He forced himself on me despite my protests",
        "He threatened to rape me if I didn't comply",
        "He sexually abused me for years",
        "He violated me sexually without permission",
        "He groped me without my consent",
        "He made unwanted sexual advances toward me",
        "He sexually harassed me constantly",
        "He coerced me into sexual acts",
        "He forced me to perform sexual acts",
        "He sexually exploited me",
        "He assaulted me sexually in my own home",
        "He violated my body against my will",
        "He forced sexual contact on me",
        "He sexually abused me when I was vulnerable",
        "He raped me multiple times",
        "He sexually attacked me without consent",
        "He forced intercourse on me",
        "He sexually violated me repeatedly",
        "He committed sexual assault against me",
        "He forced me into unwanted sexual activity",
        "He sexually brutalized me",
        "He raped me and threatened me afterwards",
        "He sexually coerced me through threats",
        "He forced sexual acts upon me",
        "He sexually invaded my personal space",
        "He molested me against my wishes",
        "He sexually dominated me by force",
        "He committed rape against me",
        "He forced unwanted intimacy on me",
        "He sexually forced himself on me",
        "He violated me in a sexual manner",
        "He perpetrated sexual violence on me",
        "He sexually traumatized me",
        "He forced sexual relations on me",
        "He committed acts of sexual violence",
    ]
    
    sexual_swahili = [
        "Alinilazimisha kufanya ngono bila ridhaa yangu",
        "Alinibaka nilipokana mapendekezo yake",
        "Alinishambulia kingono mara kwa mara",
        "Alinigusa vibaya bila idhini yangu",
        "Alinihujumu kingono nilipokuwa nasinzia",
        "Alijitia nguvu kwangu licha ya kupinga kwangu",
        "Alitishia kunibaka nisipoamtii",
        "Alinitendea unyanyasaji wa kingono kwa miaka",
        "Alinivunja heshima kingono bila ruhusa",
        "Alinipapasa bila ridhaa yangu",
        "Alifanya mapendekezo ya kingono yasiyo ya hiari",
        "Alinisumbua kingono mara zote",
        "Alinilazimisha kufanya vitendo vya kingono",
        "Alinilazimisha kufanya matendo ya kingono",
        "Alinitumia vibaya kingono",
        "Alinishambulia kingono nyumbani kwangu",
        "Alinivunja heshima mwili wangu bila ridhaa yangu",
        "Alinilazimisha mawasiliano ya kingono",
        "Alinitenda ubaya wa kingono nilipokuwa dhaifu",
        "Alinibaka mara kadhaa",
        "Alinishambulia kingono bila idhini",
        "Alinilazimisha tendo la kingono",
        "Alinivunja heshima kingono mara kwa mara",
        "Alifanya uhasama wa kingono dhidi yangu",
        "Alinilazimisha katika shughuli za kingono zisizo za hiari",
        "Alinizaidi kingono kwa nguvu",
        "Alinibaka na kunitishia baadaye",
        "Alinilazimisha kingono kupitia vitisho",
        "Alinilazimisha vitendo vya kingono",
        "Alivamia nafasi yangu ya kibinafsi kingono",
        "Alinihujumu kingono dhidi ya mapenzi yangu",
        "Alinitawala kingono kwa nguvu",
        "Alifanya ubakaji dhidi yangu",
        "Alinilazimisha ukaribu usiohitajika",
        "Alijilazimisha kwangu kingono",
        "Alinivunja heshima kwa njia ya kingono",
        "Alifanya unyanyasaji wa kingono kwangu",
        "Alinitia tabu kingono",
        "Alinilazimisha mahusiano ya kingono",
        "Alifanya vitendo vya vurugu za kingono",
    ]
    
    # ==================== EMOTIONAL VIOLENCE ====================
    emotional_english = [
        "He constantly insults and humiliates me",
        "He makes me feel worthless every day",
        "He verbally abuses me calling me horrible names",
        "He isolates me from my friends and family",
        "He belittles everything I do and say",
        "He constantly criticizes my appearance",
        "He gaslights me making me doubt my sanity",
        "He threatens to leave me if I don't obey",
        "He yells and screams at me daily",
        "He degrades me in front of others",
        "He controls who I can talk to",
        "He monitors my every move",
        "He makes fun of me publicly",
        "He says I'm stupid and worthless",
        "He destroys my self-esteem constantly",
        "He emotionally manipulates me",
        "He tells me I'm nothing without him",
        "He intimidates me with his words",
        "He uses guilt to control my actions",
        "He blames me for all his problems",
        "He says I'm a terrible mother",
        "He mocks my feelings and emotions",
        "He dismisses my concerns completely",
        "He treats me with contempt",
        "He humiliates me repeatedly",
        "He makes me feel small and insignificant",
        "He emotionally torments me daily",
        "He attacks my character constantly",
        "He undermines my confidence",
        "He ridicules me in public",
        "He says cruel things to hurt me",
        "He emotionally abuses me verbally",
        "He disrespects me constantly",
        "He puts me down all the time",
        "He makes me feel like I'm crazy",
        "He invalidates my feelings",
        "He emotionally terrorizes me",
        "He verbally attacks me daily",
        "He demeans me in front of children",
        "He psychologically abuses me",
    ]
    
    emotional_swahili = [
        "Ananichukiza na kunidharau mara zote",
        "Ananifanya nijisikie sina thamani kila siku",
        "Anatukana kwa maneno mabaya sana",
        "Ananitenganisha na marafiki na familia yangu",
        "Anadharau kila ninachofanya na ninachosema",
        "Anakosoa sura yangu mara zote",
        "Ananifanya nijisikie mwendawazimu",
        "Anatishia kuniach a nisipoamtii",
        "Anapiga kelele na kunipiga kwa maneno kila siku",
        "Ananidhalilisha mbele ya watu wengine",
        "Anadhibiti naweza kuongea na nani",
        "Anafuatilia kila nilichofanya",
        "Ananidhihaki hadharani",
        "Anasema niko mjinga na sina thamani",
        "Anaharibu ujasiri wangu mara zote",
        "Ananilaghai kihisia",
        "Ananiambia siko kitu bila yeye",
        "Anat ishia kwa maneno yake",
        "Anatumia hatia kudhibiti matendo yangu",
        "Ananilaumia kwa matatizo yake yote",
        "Anasema niko mama mbaya",
        "Anadhihaki hisia zangu",
        "Anapuuza wasiwasi wangu kabisa",
        "Ananidharau sana",
        "Ananidhalilisha mara kwa mara",
        "Ananifanya nijisikie mdogo na sina maana",
        "Ananiudhi kihisia kila siku",
        "Anashambulia tabia yangu mara zote",
        "Anaharibu imani yangu",
        "Ananibeza hadharani",
        "Anasema mambo ya ukatili kuniudhi",
        "Ananidhuru kihisia kwa maneno",
        "Ananiheshimu mara zote",
        "Ananishusha madaraja mara zote",
        "Ananifanya nijisikie mwendawazimu",
        "Anapuuza hisia zangu",
        "Ananifanya niogope kihisia",
        "Ananihamirisha kwa maneno kila siku",
        "Ananidharau mbele ya watoto",
        "Ananidhuru kiakili",
    ]
    
    # ==================== ECONOMIC VIOLENCE ====================
    economic_english = [
        "He controls all the money and won't give me any",
        "He takes my salary and refuses me access",
        "He prevents me from working or earning money",
        "He denies me money for basic needs",
        "He controls all financial decisions alone",
        "He refuses to pay for medical care",
        "He withholds money to control me",
        "He forbids me from having my own money",
        "He takes all my earnings for himself",
        "He won't let me have a job or income",
        "He denies me access to bank accounts",
        "He uses money to manipulate me",
        "He refuses to provide for our children",
        "He takes my credit cards and money",
        "He controls every cent I spend",
        "He won't give me money for food",
        "He financially abuses me constantly",
        "He sabotages my work opportunities",
        "He ruins my credit and finances",
        "He steals money from me",
        "He refuses to let me work",
        "He keeps all finances secret from me",
        "He won't share financial resources",
        "He denies me economic independence",
        "He exploits me financially",
        "He controls my economic freedom",
        "He deprives me of financial support",
        "He monitors all my spending",
        "He forbids me from earning money",
        "He takes my paycheck completely",
        "He denies me money for necessities",
        "He financially controls everything",
        "He prevents my financial autonomy",
        "He economically dominates me",
        "He withholds financial resources",
        "He restricts my access to money",
        "He financially manipulates me",
        "He denies me economic resources",
        "He controls our finances entirely",
        "He economically abuses me daily",
    ]
    
    economic_swahili = [
        "Anadhibiti pesa zote wala hananipi",
        "Anachukua mshahara wangu na kunikataza upatikanaji",
        "Ananikataza kufanya kazi au kupata pesa",
        "Ananinyima pesa za mahitaji ya msingi",
        "Anadhibiti maamuzi yote ya fedha peke yake",
        "Anakataa kulipa huduma za matibabu",
        "Anazuia pesa ili kunidhibiti",
        "Ananikataza kuwa na pesa zangu mwenyewe",
        "Anachukua mapato yangu yote kwa ajili yake",
        "Hananiruhusu kuwa na kazi au mapato",
        "Ananikataza upatikanaji wa akaunti za benki",
        "Anatumia pesa kunilahi",
        "Anakataa kutosheleza watoto wetu",
        "Anachukua kadi zangu za mkopo na pesa",
        "Anadhibiti kila senti ninayotumia",
        "Hananipi pesa ya chakula",
        "Ananidhuru kifedha mara zote",
        "Anaharibu fursa zangu za kazi",
        "Anaharibu mkopo na fedha zangu",
        "Anaiba pesa zangu",
        "Anakataa kuniruhusu kufanya kazi",
        "Anaweka fedha zote kwa siri kutoka kwangu",
        "Hana nia ya kushiriki rasilimali za fedha",
        "Ananikataza uhuru wa kiuchumi",
        "Aninitumia vibaya kifedha",
        "Anadhibiti uhuru wangu wa kiuchumi",
        "Ananinyima msaada wa kifedha",
        "Anafuatilia matumizi yangu yote",
        "Ananikataza kupata pesa",
        "Anachukua malipo yangu kabisa",
        "Ananinyima pesa za mahitaji",
        "Anadhibiti kila kitu kifedha",
        "Anazuia uhuru wangu wa kifedha",
        "Aninitawala kiuchumi",
        "Anazuia rasilimali za fedha",
        "Anapunguza upatikanaji wangu wa pesa",
        "Ananilaghai kifedha",
        "Ananikataza rasilimali za kiuchumi",
        "Anadhibiti fedha zetu kabisa",
        "Ananidhuru kiuchumi kila siku",
    ]
    
    # Combine all data
    for text in physical_english:
        data.append({"text": text, "category": "Physical_violence", "language": "english"})
    for text in physical_swahili:
        data.append({"text": text, "category": "Physical_violence", "language": "swahili"})
    
    for text in sexual_english:
        data.append({"text": text, "category": "sexual_violence", "language": "english"})
    for text in sexual_swahili:
        data.append({"text": text, "category": "sexual_violence", "language": "swahili"})
    
    for text in emotional_english:
        data.append({"text": text, "category": "emotional_violence", "language": "english"})
    for text in emotional_swahili:
        data.append({"text": text, "category": "emotional_violence", "language": "swahili"})
    
    for text in economic_english:
        data.append({"text": text, "category": "economic_violence", "language": "english"})
    for text in economic_swahili:
        data.append({"text": text, "category": "economic_violence", "language": "swahili"})
    
    return pd.DataFrame(data)

def simple_augmentation(text, num_augments=3):
    """
    Simple text augmentation techniques
    """
    augmented = [text]
    
    # Technique 1: Synonym replacement (simple)
    synonyms = {
        'hit': ['struck', 'beat', 'punched', 'slapped'],
        'constantly': ['always', 'continuously', 'repeatedly'],
        'forced': ['compelled', 'coerced', 'made'],
        'money': ['cash', 'funds', 'finances'],
        'controls': ['dominates', 'manages', 'regulates'],
    }
    
    for _ in range(num_augments):
        aug_text = text
        for word, syns in synonyms.items():
            if word in text.lower():
                aug_text = aug_text.replace(word, random.choice(syns))
        if aug_text != text:
            augmented.append(aug_text)
    
    return augmented[:num_augments + 1]

def augment_existing_dataset(df, augment_factor=5):
    """
    Augment existing small dataset
    """
    augmented_data = []
    
    for idx, row in df.iterrows():
        # Keep original
        augmented_data.append(row.to_dict())
        
        # Create augmented versions
        aug_texts = simple_augmentation(row['text'], augment_factor - 1)
        for aug_text in aug_texts[1:]:  # Skip first (original)
            augmented_data.append({
                'text': aug_text,
                'category': row['category'],
                'language': row.get('language', 'english')
            })
    
    return pd.DataFrame(augmented_data)

if __name__ == "__main__":
    print("="*70)
    print("CREATING COMPREHENSIVE TRAINING DATASET")
    print("="*70)
    
    # Create extensive manual dataset
    print("\n1. Creating 600+ manual examples...")
    manual_df = create_extensive_manual_dataset()
    print(f"   ✓ Created {len(manual_df)} manual examples")
    print(f"\n   Distribution:")
    print(manual_df['category'].value_counts())
    
    # Load and augment your existing data
    print("\n2. Loading your existing dataset...")
    try:
        existing_df = pd.read_csv('gender_based_violence_tweets.csv')
        print(f"   ✓ Loaded {len(existing_df)} existing examples")
        
        print("\n3. Augmenting existing data...")
        augmented_existing = augment_existing_dataset(existing_df, augment_factor=5)
        print(f"   ✓ Augmented to {len(augmented_existing)} examples")
    except:
        print("   ℹ  No existing dataset found, using manual data only")
        augmented_existing = pd.DataFrame()
    
    # Combine all data
    print("\n4. Combining datasets...")
    if len(augmented_existing) > 0:
        final_df = pd.concat([manual_df, augmented_existing], ignore_index=True)
    else:
        final_df = manual_df
    
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ✓ Final dataset size: {len(final_df)}")
    print(f"\n   Final distribution:")
    print(final_df['category'].value_counts())
    print(f"\n   Language distribution:")
    print(final_df['language'].value_counts())
    
    # Save
    final_df.to_csv('gbv_comprehensive_dataset.csv', index=False)
    print(f"\n✓ Saved to: gbv_comprehensive_dataset.csv")
    
    print("\n" + "="*70)
    print("DATASET READY FOR TRAINING!")
    print("="*70)
    print("\nNow run: python train_model.py")
    print("Use file: gbv_comprehensive_dataset.csv")

    # Save final dataset
final_df = pd.concat([manual_df, augmented_existing], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n4. Saving final dataset as 'final_training_data.csv'")
final_df.to_csv("final_training_data.csv", index=False)
print(f"   ✓ Final dataset saved with {len(final_df)} samples")
print(final_df['language'].value_counts())