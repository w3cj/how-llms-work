/**
 * Curated training corpus for Word2Vec Skip-gram and transformer training.
 *
 * These sentences are designed so that semantically related words frequently
 * co-occur within small windows (2-3 words apart). The corpus covers several
 * semantic clusters — animals, food, royalty, professions, properties — with
 * enough overlap that the model can discover relationships purely from
 * co-occurrence statistics.
 *
 * The royalty sentences use parallel structure ("the king is a man", "the queen
 * is a woman") so the model can learn the analogy: king - man + woman ≈ queen.
 *
 * Tokenization uses BPE (Byte Pair Encoding) trained on this corpus, connecting
 * the Skip-gram demo to the BPE tokenizer demo — the same algorithm that splits
 * text into subword tokens also defines the vocabulary for embedding training.
 */
import { applyMerges, trainBpeOnText } from "../../server/lib/bpe.js";

export const CORPUS = [
  // Animals — pets
  "the cat sat on the mat",
  "the dog sat on the rug",
  "a cat is a small pet",
  "a dog is a loyal pet",
  "the cat chased the mouse",
  "the dog chased the cat",
  "the kitten is a baby cat",
  "the puppy is a baby dog",
  "the cat and the dog are pets",
  "a kitten is small and cute",
  "a puppy is small and playful",
  "the cat sleeps on the bed",
  "the dog sleeps on the floor",
  "she loves her pet cat",
  "he loves his pet dog",
  "the cat drinks milk",
  "the dog eats meat",
  "cats and dogs are popular pets",
  "the happy cat purred loudly",
  "the happy dog wagged its tail",

  // Animals — wild
  "the lion is a wild animal",
  "the tiger is a wild animal",
  "the elephant is a big animal",
  "the mouse is a tiny animal",
  "lions and tigers are big cats",
  "the bear lives in the forest",
  "the wolf lives in the forest",
  "the eagle flies in the sky",
  "the fish swims in the water",
  "birds fly and fish swim",

  // Food
  "i ate pizza for dinner",
  "she ate pasta for lunch",
  "he ate sushi for dinner",
  "pizza and pasta are italian food",
  "sushi is japanese food",
  "bread and cheese make a sandwich",
  "fruit and vegetables are healthy food",
  "cake and cookies are sweet food",
  "rice is a common food",
  "i love pizza and pasta",
  "she cooked dinner for the family",
  "he made lunch at home",
  "the chef cooked a delicious meal",
  "coffee and tea are hot drinks",
  "juice and water are cold drinks",

  // Royalty — parallel structure for analogies
  "the king is a man who rules",
  "the queen is a woman who rules",
  "the king sits on the throne",
  "the queen sits on the throne",
  "the prince is the son of the king",
  "the princess is the daughter of the queen",
  "the king and queen rule the kingdom",
  "a prince is a young man of royal blood",
  "a princess is a young woman of royal blood",
  "the king wore a golden crown",
  "the queen wore a silver crown",
  "the prince will become king",
  "the princess will become queen",
  "the man became king of the land",
  "the woman became queen of the land",
  "the king is a powerful man",
  "the queen is a powerful woman",
  "the man was crowned king",
  "the woman was crowned queen",
  "the prince is a boy of noble birth",
  "the princess is a girl of noble birth",
  "the prince is a royal boy",
  "the princess is a royal girl",
  "the young prince played in the castle",
  "the young princess played in the castle",

  // Professions
  "the doctor works at the hospital",
  "the nurse works at the hospital",
  "the teacher works at the school",
  "the student learns at the school",
  "the chef works in the kitchen",
  "the doctor heals the sick",
  "the teacher helps the student learn",
  "the nurse helps the doctor",
  "the scientist works in the lab",
  "the engineer builds machines",

  // Properties — size
  "the elephant is very big",
  "the mouse is very small",
  "the cat is small and quick",
  "the dog is big and strong",
  "the lion is big and fierce",
  "the kitten is tiny and cute",
  "the bear is large and powerful",

  // Actions and relationships
  "the boy and the girl play together",
  "the man and the woman walk together",
  "a boy is a young man",
  "a girl is a young woman",
  "the boy will grow into a man",
  "the girl will grow into a woman",
  "the boy runs fast",
  "the girl runs fast",
  "the man is tall and strong",
  "the woman is tall and smart",
  "he is a kind man",
  "she is a kind woman",
  "the boy helped his father",
  "the girl helped her mother",

  // Nature
  "the sun shines in the sky",
  "the moon glows at night",
  "stars shine in the dark sky",
  "the river flows to the sea",
  "rain falls from the sky",
  "the tree grows in the forest",
];

export const STORIES = [
  // Royal fairy tales
  "once upon a time there was a king who ruled a great kingdom. the king was a tall and powerful man. he sat on a golden throne in a big castle. one day the king said i need a queen to rule with me. so he found a wise and kind woman and she became queen. the queen wore a silver crown and sat on the throne. they ruled the kingdom together and lived happily ever after.",

  "a young prince lived in a castle with the king and queen. the prince was a brave boy who loved to play in the garden. one day he wandered into the deep forest and found a lost princess. the princess was a clever girl of noble birth. the prince said come with me back to the castle. so the prince and the princess walked through the forest together. they returned to the kingdom and the king and queen were happy.",

  "there once was a wicked old man who took the golden crown from the king. the kingdom fell into darkness and the queen was very sad. the young prince said i will find the crown and bring it back. he went on a long journey through the forest and across the river. the brave prince found the old man in a small house by the sea. he took back the crown and returned to the castle. the king wore the crown again and the kingdom was happy once more.",

  "the princess wanted to become queen one day. she was a smart and powerful young woman. the king told her you must be kind and brave to rule. so the princess went to the village and helped the poor and the sick. she gave food to the hungry and water to the old. the people loved her and said she will be a great queen. the princess returned to the castle and the king was proud.",

  "long ago a prince and a princess from a far away kingdom came to the castle. the king and queen said you are welcome here. the prince was a strong young man and the princess was a tall young woman. they sat on the throne and ate a great meal with the royal family. the prince said this is a beautiful kingdom. the king said you may stay as long as you wish. and so they lived happily in the castle.",

  // Animal fairy tales
  "once upon a time a small cat lived in a village with an old woman. the cat was a loyal pet who sat on the mat by the door. one day the cat chased a tiny mouse into the deep forest. in the forest the cat met a big wild dog. the dog said i am lost and hungry. the kind cat said come with me back to the village. so the cat and the dog walked home together and the old woman gave them both milk and meat.",

  "there was a little kitten who wandered away from home. the kitten was small and cute but very brave. she went through the garden and into the dark forest. there she saw a great bear and a fierce wolf. the kitten was not afraid and said i am looking for my mother. the bear said your mother the cat lives by the river. so the brave little kitten found her mother and they went back home. the puppy next door wagged its tail when they returned.",

  "a loyal dog lived with a boy in a house near the forest. the dog and the boy played together every day. one day they went to the river and saw a fish swim in the water. the boy said i wish i could swim like a fish. the dog jumped into the river and the boy laughed. then an eagle flew across the sky above them. the boy and his dog walked home through the forest as the sun went down.",

  "in a kingdom by the sea there lived a cat and a dog. the cat was quick and small and the dog was big and strong. one day the king said i need a clever pet to help me find my lost crown. the cat and the dog went into the forest together. they found the golden crown under a tall tree. the king was so happy he gave them a great meal of meat and milk. the cat purred loudly and the dog wagged its tail.",

  "once there were three baby animals in the forest. a kitten a puppy and a tiny mouse. they played together near the river every day. the kitten chased the mouse and the puppy chased the kitten. one day a big lion came to the river. the three little animals were afraid but the brave puppy said we are not scared. the lion laughed and said you are small but very brave. the lion walked away and the three friends played happily.",

  "the old woman had a cat who loved to sleep on the bed. one night the cat heard a sound at the door. she went outside and saw a lost baby bird in the garden. the cat was kind and did not chase the bird. she said come inside it is cold and dark. the bird slept on the mat and in the morning the cat took the bird to the tall tree by the river. the bird flew into the sky and the cat went home.",

  // Village and profession tales
  "once upon a time there was a doctor who worked at the hospital in a small village. one day a sick boy came to the door. the doctor said i will help you. the nurse helped the doctor and they healed the boy. the boy said thank you and went home to his family. his mother made him a warm meal of bread and cheese. the boy ate his dinner and slept on his bed.",

  "a clever young woman became a teacher at the school in the village. she helped every student learn to read and write. one day a boy said i want to become a doctor. the teacher said you must study hard and be brave. the boy worked every day at the school and the teacher was proud. he grew into a tall strong man and went to work at the hospital. the teacher said i am happy i could help.",

  "the chef worked in the kitchen of the castle. he cooked a delicious meal for the king and queen every day. one day the king said i want pizza and pasta for dinner. the chef said but pizza is italian food and pasta is italian food too. the king laughed and said i love italian food. so the chef made the best pizza and pasta in the kingdom. the queen ate sushi and rice because she loved japanese food.",

  "there was a scientist who worked in a lab near the forest. the scientist was a clever old man who loved animals. one day he found a sick baby bear by the river. he took the bear to the doctor at the hospital. the nurse helped the doctor heal the baby bear. when the bear was strong again the scientist took him back to the forest. the bear was happy and the scientist smiled.",

  "a young man wanted to become an engineer. he went to the school in the village and studied hard. the teacher said you are a smart student. one day the engineer built a great machine. the king heard about the machine and said come to the castle. the engineer went to the castle and showed the king his machine. the king said you are a clever man and gave him a golden crown. the engineer lived happily in the kingdom.",

  // Nature and journey tales
  "once upon a time the sun and the moon had a race across the sky. the sun shines bright in the day and the moon glows at night. the sun said i am faster than you. the moon said but the stars shine with me. they raced from the river to the sea. the sun ran fast but the moon was clever and took a path through the dark sky. in the end they said we are both great and they lived happily together.",

  "a brave girl wandered into the deep forest one day. the tall trees grew thick and the sky turned dark. rain fell from the sky and the river flowed fast. the girl found a small house by the river. an old woman lived there with a loyal cat and a playful dog. the old woman said come inside and have some food. she gave the girl bread and milk and the girl slept by the fire. in the morning the girl returned home to her family.",

  "there once was a great tree that grew in the middle of the forest. birds lived in the tree and a bear slept under it. the river flowed nearby and fish swam in the water. one day a young boy found the tree and said this is a beautiful place. he came back every day and the animals were not afraid. the eagle flew down from the sky and sat on the boy his arm. the boy and the animals became friends and lived happily in the forest.",

  "long ago rain did not fall from the sky and the river was dry. the animals in the forest were very hungry and sad. the lion said we must find water. the wolf said i know a path to the sea. so the lion the wolf and the bear went on a long journey. they walked through the forest and over the land. at last they found a great river that flowed to the sea. the animals drank the cold water and were happy.",

  "the moon glowed bright one night and the stars shined in the dark sky. a little girl sat by the river and looked up. she said i wish i could fly like an eagle. then a great bird came down from the sky and said climb on my back. the girl flew over the forest and the river and the sea. she saw the king castle and the small village below. the bird took her home and she told her mother about the journey. her mother said that is a beautiful story.",

  // Food and feast tales
  "the king and queen had a great feast at the castle. the chef cooked pizza and pasta and sushi and rice. there was bread and cheese and cake and cookies on the table. the prince ate meat and the princess drank cold juice. the boy and the girl from the village came to the feast. they ate fruit and vegetables and sweet food. coffee and tea and cold water were the drinks. the king said this is the best meal in the kingdom.",

  "once there was a poor old woman who had no food. she was very hungry and sad. one day a kind man came to her door with bread and cheese. he said i am a chef and i made this food for you. the old woman ate the bread and cheese and was happy. the next day the chef came back with pasta and rice and cake. the old woman said you are the most kind man i have ever met. they became friends and ate dinner together every day.",

  "a boy and a girl went to the forest to find food for their family. they found fruit on the trees and vegetables in a garden. the girl said we should bring some back for mother and father. they walked home through the forest with the food. their mother cooked a delicious meal of rice and meat. the family sat together and ate dinner. the father said you are brave and clever children. they lived happily in their small house by the river.",

  // Mixed theme tales
  "once upon a time a young prince had a pet cat and a pet dog. the cat was small and quick and the dog was big and loyal. one day the prince took his pets to the forest. the cat chased a mouse and the dog chased the cat. the prince laughed and said you two are very playful. they sat by the river and the prince ate bread and cheese for lunch. the sun shined in the sky and the prince and his pets were happy.",

  "there was a princess who loved animals. she had a kitten and a puppy in the castle. one day she found a baby bird in the garden. she asked the doctor to help the sick bird. the doctor and the nurse healed the bird. the princess was happy and gave the doctor a golden crown. the kitten and the puppy played with the bird in the castle. the king and queen said our daughter is a kind and brave girl.",

  "long ago a wise old man lived in a house by the sea. he had a loyal dog and a clever cat. every day he sat on a mat by the door and watched the sun shine over the water. one day a young boy came to his door and said i am lost. the old man gave the boy food and water and said you can stay here. the boy lived with the old man and helped him every day. the cat slept on the bed and the dog slept on the floor. they all lived happily together.",

  "a brave young woman left the village and went on a journey to find the lost kingdom. she walked through the deep forest and across the great river. she met a lion who said i will help you. they found the kingdom hidden behind a tall tree. the old castle was dark and a wicked man sat on the throne. the brave woman and the lion chased the wicked man away. she became queen and ruled the kingdom with kindness. the lion lived in the castle garden and they were happy ever after.",

  "once upon a time in a small village there lived a boy and a girl. the boy wanted to become a king and the girl wanted to become a queen. they went to the castle and asked the old king for help. the king said you must be brave and kind and smart. the boy and the girl went on a long journey through the forest. they helped the animals and the poor people they found. when they returned the king said you are ready. the boy became king and the girl became queen and they ruled the kingdom together.",

  // Additional tales — covering remaining vocabulary
  "the king had a large elephant that lived in the castle garden. the elephant was a big and powerful animal. one day a fierce tiger came from the wild forest. the tiger runs fast and the elephant was afraid. the brave prince said i will help you. the lion and the wolf came to help too. the tiger saw the lions and tigers together and ran away. the elephant was happy and the prince was crowned a hero. the king sits on the throne and rules the kingdom with the prince who will grow into a great man.",

  "every morning the boy runs to the school in the village. his dog runs with him and they are both very fast. the teacher helps the student learn and the nurse helps the doctor at the hospital. the chef works in the kitchen and makes a common meal of rice and bread. hot coffee and tea are popular drinks in the village. the boy eats his lunch of meat and a sandwich and drinks cold juice. then he walks home and his cat sleeps on the rug by the door. his dogs sit on the floor and make a loud sound.",

  "the old man and the old woman loved to walk by the river. the river flows to the sea and the tree grows on the land beside it. birds fly in the sky and the eagle flies above them all. the fish swims in the water and the bear lives in the forest nearby. the old woman loves her pet cats and dogs. they are popular pets in the village. the old man builds machines and works as an engineer. their son learns at the school and their daughter heals the sick at the hospital.",

  "a baby elephant and a baby tiger played in the forest. the elephant was big and the tiger was fierce but they were friends. the elephant eats fruit and vegetables which are healthy food. the tiger eats meat because it is a wild animal. one day rain falls from the sky and the river flows very fast. the two animals sat under a large tree and the bear sat with them. the wolf sat nearby and the eagle sat in the tall tree above. when the sun came back they all played happily together.",

  "the princess had a kitten that sleeps on her bed and a puppy that sleeps on the floor. the kitten is tiny and cute and the puppy is small and playful. every day the princess walks with her pets in the garden. the kitten runs and the puppy runs after her. they are both popular pets in the kingdom. the queen loves her daughter and said she grows into a brave young woman. the king said our daughter is of royal blood and noble birth. she is a true animal lover. one day she will rule this land.",
];

const BPE_MERGE_COUNT = 500;
const { merges } = trainBpeOnText(CORPUS.join(" ").toLowerCase(), BPE_MERGE_COUNT);
/** Pre-trained BPE merge table — computed once at import time from the full corpus. */
export const BPE_MERGES = merges;

/** Tokenize text using the corpus BPE merges — lowercases first, then applies all learned merges. */
export function tokenize(text: string): string[] {
  return applyMerges(text.toLowerCase(), BPE_MERGES);
}

/** Build a vocabulary from the corpus — returns word↔index mappings sorted by frequency (most common first). */
export function buildVocab(corpus: string[]) {
  const freq = new Map<string, number>();
  for (const sentence of corpus) {
    for (const word of tokenize(sentence)) {
      freq.set(word, (freq.get(word) ?? 0) + 1);
    }
  }

  const indexToWord: string[] = [];
  const wordToIndex = new Map<string, number>();

  for (const [word] of [...freq.entries()].sort((a, b) => b[1] - a[1])) {
    wordToIndex.set(word, indexToWord.length);
    indexToWord.push(word);
  }

  return { wordToIndex, indexToWord, freq };
}
