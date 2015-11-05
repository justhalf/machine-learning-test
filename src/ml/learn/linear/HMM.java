package ml.learn.linear;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;

/**
 * A class implementing simple HMM
 * This HMM is specialized for learning tags of a word sequence, as it uses some morphology
 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
 *
 */
public class HMM implements StructuredClassifier{
	
	private static boolean DEBUG = false;
	
	/** Indicating start of the sequence */
	public static final Tag START = Tag.get("START");
	/** Indicating end of the sequence */
	public static final Tag END = Tag.get("END");
	
	/** Indicate an unknown word */
	public static final String UNKNOWN_WORD = "-UNK-";
	/** Indicate a numeric */
	public static final String NUMERIC = "-NUM-";
	public static final String[] INIT_CAP_FEATURES = {"##NOINIT_NOCAP##", "##NOINIT_CAP##", "##INIT_NOCAP##", "##INIT_CAP##"};
	public static final String[] END_FEATURES = {"##END_S##", "##END_ED##", "##END_ING##", "##END_OTHER##"};
	
	/** The mapping from tags to tag indices */
	public LinkedHashMap<Tag, Integer> tags;
	/** An array for easy mapping from tag indices to tags */
	public Tag[] reverseTags;
	/** The mapping from words to word indices */
	public LinkedHashMap<String, Integer> features;
	/** An array for easy mapping from word indices to words */
	public String[] reverseFeatures;
	
	/** Transition probability */
	public double[][] transitionProbs;
	/** Word emission probability */
	public double[][] tagFeatureProbs;
	
	/** The smoothing method to be used */
	public SmoothingMethod smoothingMethod;
	
	/** To enable closed class tags */
	public boolean enableClosedClass;
	
	/** To use morphology features */
	public boolean useMorphologyFeatures;
	
	/** To reduce smoothing effect */
	public boolean reduceSmoothingEffect;
	
	/** To use hapax legomena to estimate unknown word count per tag */
	public boolean useHapaxLegomena;
	
	/**
	 * Enumeration of possible smoothing method
	 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
	 *
	 */
	public static enum SmoothingMethod{
		NONE, ADD_ONE
	}
	
	/**
	 * Initialize HMM with default settings
	 */
	public HMM(){
		smoothingMethod = SmoothingMethod.ADD_ONE;
		reduceSmoothingEffect = true;
		enableClosedClass = false;
		useMorphologyFeatures = false;
		useHapaxLegomena = true;
	}
	
	public Map<Tag, Integer> getTags(){
		return tags;
	}
	
	@Override
	public void train(List<Instance> trainingData){
		reset();
		getNumTagsAndWords(trainingData);
		transitionProbs = new double[tags.size()-1][tags.size()];
		tagFeatureProbs = new double[tags.size()][features.size()];
		reverseTags = new Tag[tags.size()];
		for(Tag tag: tags.keySet()){
			reverseTags[tags.get(tag)] = tag;
		}
		reverseFeatures = new String[features.size()];
		for(String word: features.keySet()){
			reverseFeatures[features.get(word)] = word;
		}
		doCounting(trainingData);
		if(DEBUG){
			writeTagTagCount();
			writeWordTagCount();
		}
		determineOpenClasses((int)Math.round(features.size()/500.0));
		doSmoothing();
		calculateProb();
		if(DEBUG){
			writeTransitionProb();
		}
	}
	
	/**
	 * Forget the probability that has already calculated
	 */
	public void reset(){
		tags = new LinkedHashMap<Tag, Integer>();
		features = new LinkedHashMap<String, Integer>();
		transitionProbs = null;
		tagFeatureProbs = null;
	}
	
	/**
	 * First pass of the training data to get the number of words and tags
	 * @param trainingData
	 */
	private void getNumTagsAndWords(List<Instance> trainingData){
		for(Instance instance: trainingData){
			for(TaggedWord wordTag: instance.words){
				String word = wordTag.word();
				Tag tag = wordTag.tag();
				word = normalize(word, true);
				if(!features.containsKey(word)){
					features.put(word, features.size());
				}
				if(!tags.containsKey(tag)){
					tags.put(tag, tags.size());
				}
			}
		}
		tags.put(START, tags.size());
		tags.put(END, tags.size());
		features.put(UNKNOWN_WORD, features.size());
		
		for(String feature: INIT_CAP_FEATURES){
			features.put(feature, features.size());
		}
		for(String feature: END_FEATURES){
			features.put(feature, features.size());
		}
	}
	
	/**
	 * Second pass of the training data to count the n-grams required
	 * @param trainingData
	 */
	private void doCounting(List<Instance> trainingData){
		for(Instance instance: trainingData){
			int prevTagIdx = tags.get(START);
			boolean firstWord = true;
			for(TaggedWord wordTag: instance.words){
				String word = wordTag.word();
				Tag tag = wordTag.tag();
				int initCapFeature = getInitCapFeature(word, firstWord);
				int endFeature = getEndFeature(word);
				word = normalize(word, true);
				int featureIdx = features.get(word);
				int tagIdx = tags.get(tag);
				transitionProbs[prevTagIdx][tagIdx] += 1;
				tagFeatureProbs[tagIdx][featureIdx] += 1;
				if(tag.isOpenClass){
					tagFeatureProbs[tagIdx][initCapFeature] += 1;
					tagFeatureProbs[tagIdx][endFeature] += 1;
				}
				prevTagIdx = tagIdx;
				firstWord = false;
			}
			int tagIdx = tags.get(END);
			transitionProbs[prevTagIdx][tagIdx] += 1;
		}
	}
	
	/**
	 * Returns the index to the feature representing whether the word is sentence initial 
	 * and whether it starts with a capital letter
	 * @param word
	 * @param firstWord
	 * @return
	 */
	private int getInitCapFeature(String word, boolean firstWord){
		boolean isCapital = false;
		if(word.matches("[A-Z].*")){
			isCapital = true;
		}
		return features.get(INIT_CAP_FEATURES[2*(firstWord ? 1 : 0)+(isCapital ? 1 : 0)]);
	}
	
	/**
	 * Returns the index to the feature representing whether the word ends in "s", "ed", "ing", or others.
	 * @param word
	 * @return
	 */
	private int getEndFeature(String word){
		word = word.toLowerCase();
		if(word.matches(".*s")) return features.get(END_FEATURES[0]);
		else if (word.matches(".*ed")) return features.get(END_FEATURES[1]);
		else if (word.matches(".*ing")) return features.get(END_FEATURES[2]);
		else return features.get(END_FEATURES[3]);
	}
	
	/**
	 * A method to automatically determine whether certain tags can emit unknown words
	 * @param limit
	 */
	private void determineOpenClasses(int limit){
		for(int tagIdx=0; tagIdx<tags.size(); tagIdx++){
			Set<String> wordsSet = new HashSet<String>();
			String sample = "";
			for(int featureIdx=0; featureIdx<features.size()-8; featureIdx++){
				if(tagFeatureProbs[tagIdx][featureIdx] > 0){
					sample = reverseFeatures[featureIdx];
					wordsSet.add(sample.toLowerCase());
				}
			}
//			System.out.println("The tag "+reverseTags[tagIdx].text+" has "+wordsSet.size()+" unique entries! (e.g., "+sample+")");
			if(wordsSet.size() <= limit){
				reverseTags[tagIdx].isOpenClass = false;
//				System.out.println("The tag "+reverseTags[tagIdx].text+" is closed!");
			}
		}
	}
	
	/**
	 * Normalize the word to the canonical representation (e.g., all numbers to a single token)
	 * @param word
	 * @param isTraining
	 * @return
	 */
	private String normalize(String word, boolean isTraining){
		int digits = 0;
		for(char c: word.toCharArray()){
			if(c >= 48 && c <= 57){
				digits++;
			}
		}
		if(word.matches("[0-9]+([0-9,-.])*") || digits*3 >= word.length()*2){
			word = NUMERIC;
		}
		if(!isTraining && !features.containsKey(word)) return UNKNOWN_WORD;
		return word;
	}
	
	/**
	 * Perform smoothing according to the {@link #smoothingMethod} variable
	 */
	private void doSmoothing(){
		if(smoothingMethod == SmoothingMethod.ADD_ONE){
			// Smoothing on transition probability estimate counts
			for(int tagIdx1=0; tagIdx1<tags.size()-1; tagIdx1++){
				for(int tagIdx2=0; tagIdx2<tags.size(); tagIdx2++){
					if(tagIdx2 != tags.get(START)){
						transitionProbs[tagIdx1][tagIdx2] += 1;	
					}
				}
			}
			
			/*
			 * Estimate the probability tag j producing the word i p(x_i | y_j)
			 * using the probability estimation of tag j producing words that only appear once
			 */
			if(useHapaxLegomena){
				double[] uniqueCount = new double[tags.size()];
				Arrays.fill(uniqueCount, 0);
				for(int featureIdx=0; featureIdx<features.size()-8; featureIdx++){
					if(featureIdx != features.get(UNKNOWN_WORD)){
						int sum = 0;
						int idx = -1;
						for(int tagIdx=0; tagIdx<tags.size(); tagIdx++){
							sum += tagFeatureProbs[tagIdx][featureIdx];
							if(tagFeatureProbs[tagIdx][featureIdx] == 1){
								idx = tagIdx;
							}
						}
						if(sum == 1){ // Word only appears once
							uniqueCount[idx] += 1;
						}
					} else {
						for(int tagIdx=0; tagIdx<tags.size(); tagIdx++){
							tagFeatureProbs[tagIdx][featureIdx] += uniqueCount[tagIdx];
						}
					}
				}
			}
			
			// Add one for all words (including additional features)
			for(int tagIdx=0; tagIdx<tags.size(); tagIdx++){
				if(enableClosedClass && !reverseTags[tagIdx].isOpenClass){
					continue;
				}
				for(int featureIdx=0; featureIdx<features.size(); featureIdx++){
					if(reduceSmoothingEffect){
						tagFeatureProbs[tagIdx][featureIdx] *= 256; // To reduce the effect of the add-one
					}
					tagFeatureProbs[tagIdx][featureIdx] += 1;
				}
			}
		}
	}
	
	/**
	 * Calculate the probability from the counts
	 */
	private void calculateProb(){
		String[] wordFeatures = new String[features.size()-8];
		for(int i=0; i<wordFeatures.length; i++){
			wordFeatures[i] = reverseFeatures[i];
		}
		calculateProbForFeatures(wordFeatures);
		calculateProbForFeatures(INIT_CAP_FEATURES);
		calculateProbForFeatures(END_FEATURES);
		calculateProbForTransitions();
	}

	private void calculateProbForTransitions() {
		for(int tagIdx1=0; tagIdx1<tags.size()-1; tagIdx1++){
			int sum = 0;
			for(int tagIdx2=0; tagIdx2<tags.size(); tagIdx2++){
				sum += transitionProbs[tagIdx1][tagIdx2];
			}
			for(int tagIdx2=0; tagIdx2<tags.size(); tagIdx2++){
				transitionProbs[tagIdx1][tagIdx2] /= sum;
				transitionProbs[tagIdx1][tagIdx2] = Math.log(transitionProbs[tagIdx1][tagIdx2]);
			}
		}
	}
	
	private void calculateProbForFeatures(String[] featureNames){
		for(int tagIdx=0; tagIdx<tags.size(); tagIdx++){
			int sum = 0;
			for(String feature: featureNames){
				int featureIdx = features.get(feature);
				sum += tagFeatureProbs[tagIdx][featureIdx];
			}
			for(String feature: featureNames){
				int featureIdx = features.get(feature);
				tagFeatureProbs[tagIdx][featureIdx] /= sum;
				tagFeatureProbs[tagIdx][featureIdx] = Math.log(tagFeatureProbs[tagIdx][featureIdx]);
			}
		}
	}
	
	@Override
	public List<Instance> predict(List<Instance> testData){
		double[] prevProbs = new double[tags.size()-2];
		double[] curProbs = new double[tags.size()-2];
		Arrays.fill(prevProbs, 0);
		List<Instance> results = new ArrayList<Instance>();
		for(Instance instance: testData){
			int[][] parentIdx = new int[tags.size()-2][instance.words.size()];
			int wordCount = 0;
			String[] wordList = new String[instance.words.size()];
			for(TaggedWord wordTag: instance.words){
				String word = wordTag.word();
				wordList[wordCount] = word;
				word = normalize(word, false);
				if(DEBUG && word == UNKNOWN_WORD){
					System.out.println("Unknown word: "+wordTag.word());
				}
				int featureIdx = features.get(word);
//				double maxCur = Double.NEGATIVE_INFINITY;
//				int maxCurIdx = -1;
				for(int tagIdx=0; tagIdx<curProbs.length; tagIdx++){
					// Word emission probability p(x_i | y_j), i=featureIdx, j=tagIdx
					double wordTagProb = tagFeatureProbs[tagIdx][featureIdx];
					if(word == UNKNOWN_WORD){
						if(useMorphologyFeatures){
							double initCapProb = tagFeatureProbs[tagIdx][getInitCapFeature(wordTag.word(), wordCount==0)];
							double endProb = tagFeatureProbs[tagIdx][getEndFeature(wordTag.word())];
							wordTagProb = wordTagProb + initCapProb + endProb;
						}
					}
					if(wordCount == 0){
						// Transition probability from START to tagIdx p(y_j | y_i), i=START, j=tagIdx
						curProbs[tagIdx] = transitionProbs[tags.get(START)][tagIdx]+wordTagProb;
						parentIdx[tagIdx][wordCount] = tags.get(START);
					} else {
						// Find the tag that gives maximum probability up to this point
						// max_i(p(y_j | y_i)), i=prevTagIdx, j=tagIdx
						curProbs[tagIdx] = Double.NEGATIVE_INFINITY;
						for(int prevTagIdx=0; prevTagIdx<prevProbs.length; prevTagIdx++){
							double prob = prevProbs[prevTagIdx]+transitionProbs[prevTagIdx][tagIdx]+wordTagProb;
							if(prob > curProbs[tagIdx]){
								curProbs[tagIdx] = prob;
								parentIdx[tagIdx][wordCount] = prevTagIdx;
							}
						}
					}
//					if(curProbs[tagIdx] > maxCur){
//						maxCur = curProbs[tagIdx];
//						maxCurIdx = tagIdx;
//					}
				}
//				System.out.println("For word: "+reverseWords[featureIdx]);
//				for(int tagIdx=0; tagIdx<curProbs.length; tagIdx++){
//					String pre = "";
//					if(tagIdx == maxCurIdx) pre = "**";
//					System.out.print(pre+reverseTags[parentIdx[tagIdx][wordCount]]+"-"+reverseTags[tagIdx]+pre+" "+String.format("%.2f ", curProbs[tagIdx]));
//				}
//				System.out.println();
				for(int i=0; i<curProbs.length; i++){
					prevProbs[i] = curProbs[i];
				}
				wordCount++;
			}
			int curIdx = -1;
			double max = Double.NEGATIVE_INFINITY;
			for(int i=0; i<curProbs.length; i++){
				if(curProbs[i]+transitionProbs[i][tags.get(END)] > max){
					curIdx = i;
					max = curProbs[i]+transitionProbs[i][tags.get(END)];
				}
			}
			List<TaggedWord> result = new ArrayList<TaggedWord>();
			for(wordCount = wordCount-1; wordCount >= 0; wordCount--){
				result.add(0, new TaggedWord(wordList[wordCount], reverseTags[curIdx]));
				curIdx = parentIdx[curIdx][wordCount];
			}
			results.add(new Instance(result));
		}
		return results;
	}

	private void writeTagTagCount() {
		try {
			FileWriter wr = new FileWriter("tagTagCount");
			wr.write(String.format("%5s ", ""));
			for(int j=0; j<transitionProbs[0].length; j++){
				wr.write(String.format("%5s ", reverseTags[j]));
			}
			wr.write("\n");
			for(int i=0; i<transitionProbs.length; i++){
				wr.write(String.format("%5s ", reverseTags[i]));
				for(int j=0; j<transitionProbs[0].length; j++){
					wr.write(String.format("%5d ", Math.round(transitionProbs[i][j])));
				}
				wr.write("\n");
			}
			wr.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void writeWordTagCount() {
		try {
			FileWriter wr = new FileWriter("wordTagCount");
			wr.write(String.format("%30s ", ""));
			for(int tagIdx=0; tagIdx<tagFeatureProbs.length; tagIdx++){
				wr.write(String.format("%5s ", reverseTags[tagIdx]));
			}
			wr.write("\n");
			for(int featureIdx=0; featureIdx<tagFeatureProbs[0].length; featureIdx++){
				wr.write(String.format("%30s ", reverseFeatures[featureIdx]));
				for(int tagIdx=0; tagIdx<tagFeatureProbs.length; tagIdx++){
					wr.write(String.format("%5d ", Math.round(tagFeatureProbs[tagIdx][featureIdx])));
				}
				wr.write("\n");
			}
			wr.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void writeTransitionProb() {
		try {
			FileWriter wr = new FileWriter("tagTagProb");
			wr.write(String.format("%5s ", ""));
			for(int j=0; j<transitionProbs[0].length; j++){
				wr.write(String.format("%9s ", reverseTags[j]));
			}
			wr.write("\n");
			for(int i=0; i<transitionProbs.length; i++){
				wr.write(String.format("%5s ", reverseTags[i]));
				for(int j=0; j<transitionProbs[0].length; j++){
					wr.write(String.format("%9.3f ", transitionProbs[i][j]));
				}
				wr.write("\n");
			}
			wr.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
