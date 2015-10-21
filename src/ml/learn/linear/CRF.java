package ml.learn.linear;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;

import lbfgsb.DifferentiableFunction;
import lbfgsb.FunctionValues;
import lbfgsb.IterationFinishedListener;
import lbfgsb.LBFGSBException;
import lbfgsb.Minimizer;
import lbfgsb.Result;
import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;
import ml.learn.optimizer.GradientDescentMinimizer;
import ml.learn.optimizer.GradientDescentMinimizer.LearnType;

public class CRF implements StructuredClassifier{
	
	/** Indicating start of the sequence */
	public static final Tag START = Tag.START;
	/** Indicating end of the sequence */
	public static final Tag END = Tag.END;
	
	/** Indicate an unknown word */
	public static final String UNKNOWN_WORD = "-UNK-";
	/** Indicate a numeric */
	public static final String NUMERIC = "-NUM-";
	public static final String[] INIT_CAP_FEATURES = {"##NOINIT_NOCAP##", "##NOINIT_CAP##", "##INIT_NOCAP##", "##INIT_CAP##"};
	public static final String[] END_FEATURES = {"##END_S##", "##END_ED##", "##END_ING##", "##END_OTHER##"};
	
	public double[] weights;
	public double regularizationParameter;
	
	public LinkedHashMap<Tag, Integer> tags;
	public Tag[] reverseTags;
	
	public LinkedHashMap<String, Integer> words;
	public String[] reverseWords;
	
	public Map<String,Integer> featureIndices;
	public static Map<Integer,String> reverseFeatureIndices;
	public Template[] templates;
	public Set<Integer>[][] tagTagCache;
	
	public int[] noStart;
	public int[] noEnd;
	public int[] onlyStart;
	public int[] onlyEnd;
	public int[] empty;
	
	public Random random;
	
//	boolean useSGD = false;
	boolean useSGD = true;
	boolean useLogSpace = true;
//	private int cnt = 0;
	//private LearnType learnType = LearnType.STOCHASTIC; 
	private LearnType learnType = LearnType.BATCH;
	private double learningRate = 0.2;
	private int iterations = 50;
	private int batchSize = 5;
	
	public CRF(){
		this(new String[]{
//				"U00:%x[-2,0]",
//				"U01:%x[-1,0]",
				"U02:%x[0,0]",
//				"U03:%x[1,0]",
//				"U04:%x[2,0]",
//				"U05:%x[-1,0]/%x[0,0]",
//				"U06:%x[0,0]/%x[1,0]",
//
//				"U10:%x[-2,1]",
//				"U11:%x[-1,1]",
				"U12:%x[0,1]",
//				"U13:%x[1,1]",
//				"U14:%x[2,1]",
//				"U15:%x[-2,1]/%x[-1,1]",
//				"U16:%x[-1,1]/%x[0,1]",
//				"U17:%x[0,1]/%x[1,1]",
//				"U18:%x[1,1]/%x[2,1]",
//
//				"U20:%x[-2,1]/%x[-1,1]/%x[0,1]",
//				"U21:%x[-1,1]/%x[0,1]/%x[1,1]",
//				"U22:%x[0,1]/%x[1,1]/%x[2,1]",
				
				"B",
				});
	}
	
	public CRF(String templateFile) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(templateFile), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		List<Template> templates = new ArrayList<Template>();
		while(br.ready()){
			String line = br.readLine().trim();
			if(line.length() == 0 || line.startsWith("#")) continue;
			templates.add(new Template(line));
		}
		br.close();
		initialize(templates.toArray(new Template[templates.size()]));
	}
	
	public CRF(String[] templates){
		Template[] templateObjects = new Template[templates.length];
		for(int i=0; i<templates.length; i++){
			templateObjects[i] = new Template(templates[i]);
		}
		initialize(templateObjects);
	}
	
	private void initialize(Template[] templates){
		random = new Random(0);
		tags = new LinkedHashMap<Tag, Integer>();
		words = new LinkedHashMap<String, Integer>();
		regularizationParameter = 1.0;
		this.templates = templates;
	}
	
	private Result minimize(DifferentiableFunction function, double[] startingPoint) throws LBFGSBException{
		long startTime = System.currentTimeMillis();
		Minimizer alg = new Minimizer();
		alg.setNoBounds(startingPoint.length);
		alg.setIterationFinishedListener(new IterationFinishedListener(){
			
			public int i=0;

			@Override
			public boolean iterationFinished(double[] point, double functionValue, double[] gradient) {
				i++;
//				System.out.println("Iteration "+i+": "+(-functionValue));
				System.out.printf("Iteration %d: %.3f, %.3fs elapsed\n", i, (-functionValue), (System.currentTimeMillis()-startTime)/1000.0);
				gradient = negate(gradient);
//				System.out.println("Point:");
//				for(int j=0; j<point.length; j++){
//					System.out.printf("%.4f ", point[j]);
//				}
//				System.out.println();
//				System.out.println("Gradient:");
//				for(int j=0; j<point.length; j++){
//					System.out.printf("%.4f ", point[j]);
//				}
//				System.out.println();
				return true;
			}
			
		});
		return alg.run(function, startingPoint);
	}
	
	private static class AccumulatorResult{
		public double value;
		public int maxIdx;
		
		public AccumulatorResult(double value, int maxIdx){
			this.value = value;
			this.maxIdx = maxIdx;
		}
		
		public AccumulatorResult(double value){
			this(value, -1);
		}
	}
	
	/**
	 * Return the sum of values in the given double array, ignoring NaN
	 * @param values
	 * @return
	 */
	private static AccumulatorResult sum(double[] values){
		double result = 0;
		for(double value: values){
			if(Double.isNaN(value)) continue;
			result += value;
		}
		return new AccumulatorResult(result);
	}
	
	/**
	 * Return the maximum value in the given double array, ignoring NaN.
	 * Will also set the index of the maximum value in the array
	 * @param values
	 * @return
	 */
	private static AccumulatorResult max(double[] values){
		double result = Double.NEGATIVE_INFINITY;
		int parentIdx = -1;
		for(int i=0; i<values.length; i++){
			if(Double.isNaN(values[i])) continue;
			if(values[i] > result){
				result = values[i];
				parentIdx = i;
			}
		}
		return new AccumulatorResult(result, parentIdx);
	}
	
	private static AccumulatorResult logSumOfExponentials(double[] xs) {
        if (xs.length == 1) return new AccumulatorResult(xs[0]);
        double max = max(xs).value;
        double sum = 0.0;
        for (int i = 0; i < xs.length; ++i) {
        	if(Double.isNaN(xs[i])) continue;
            if (xs[i] != Double.NEGATIVE_INFINITY)
                sum += java.lang.Math.exp(xs[i] - max);
        }
        return new AccumulatorResult(max + java.lang.Math.log(sum));
    }
	
	/**
	 * A loglikelihood function
	 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
	 *
	 */
	public class LogLikelihood implements DifferentiableFunction{
		
		public List<Instance> trainingData;
		public List<Instance> staticTrainingData;
		public LinkedHashMap<Instance, double[][]> forwards;
		public LinkedHashMap<Instance, double[][]> backwards;
		public double[] empiricalDistribution;
		
		public LogLikelihood(List<Instance> trainingData){
			this.staticTrainingData = trainingData;
			this.trainingData = trainingData;
			empiricalDistribution = computeEmpiricalDistribution();
		}

		private double[] computeEmpiricalDistribution(){
			double[] result = new double[featureIndices.size()];
			Arrays.fill(result, 0);
			int size = trainingData.size();
			int total = 0;
			for(Instance instance: trainingData){
				int n = instance.words.size()+2;
				for(int j=0; j<n-1; j++){
					Tag prevTag = instance.getTagAt(j-1);
					Tag curTag = instance.getTagAt(j);
					for(int i: featuresPresent(instance, j, prevTag, curTag)){
						if(i == -1) continue;
						result[i] += 1;
					}
				}
				total++;
				if(total % 10000 == 0){
					System.out.println(String.format("Completed %d/%d", total, size));
				}
			}
			return result;
		}
		
		@Override
		public FunctionValues getValues(double[] point){
			return getValues(point, staticTrainingData);
		}

		public FunctionValues getValues(double[] point, List<Instance> trainingData) {
			this.trainingData = trainingData;
			empiricalDistribution = computeEmpiricalDistribution();
//			System.out.println("Computing forward backward...");
			long startTime = System.currentTimeMillis();
			computeForwardBackward(point);
			long endTime = System.currentTimeMillis();
//			System.out.printf("Done forward-backward in %.3fs\n", (endTime-startTime)/1000.0);
			double value = 0.0;
			startTime = System.currentTimeMillis();
			for(Instance instance: trainingData){
				int n = instance.words.size()+2;
				for(int position=0; position<n-1; position++){
					Tag prevTag = instance.getTagAt(position-1);
					Tag curTag = instance.getTagAt(position);
					for(int i: featuresPresent(instance, position, prevTag, curTag)){
						if(i == -1) continue;
						value += point[i];
					}
				}
				if (useLogSpace) {
					value -= normalizationConstant(instance);
				} else {
					value -= Math.log(normalizationConstant(instance));
				}
			}
			value -= regularizationConstant(point);
			endTime = System.currentTimeMillis();
//			System.out.printf("Done calculating value in %.3fs\n", (endTime-startTime)/1000.0);
			startTime = System.currentTimeMillis();
			double[] gradient = computeGradient(point);
			endTime = System.currentTimeMillis();
//			System.out.printf("Done calculating gradient in %.3fs\n", (endTime-startTime)/1000.0);
			
//			cnt++;
//			if (cnt % 10 == 0) {
//				System.out.println("Function calls:" + cnt);
//			}
			
			// Values and gradient are negated to find the maximum
			return new FunctionValues(-value, negate(gradient));
		}
		
		/**
		 * Compute lambda^2/(2.sigma^2) for calculating likelihood value
		 * @return
		 */
		private double regularizationConstant(double[] point){
			double result = 0;
			for(int i=0; i<point.length; i++){
				result += Math.pow(point[i], 2);
			}
			result /= 2*Math.pow(regularizationParameter, 2);
			return result;
		}
		
		private void computeForwardBackward(double[] point){
			forwards = new LinkedHashMap<Instance, double[][]>();
			backwards = new LinkedHashMap<Instance, double[][]>();
//			int size = trainingData.size();
//			int total = 0;
			for(Instance instance: trainingData){
				int n = instance.words.size()+2;
				double[][] forward = new double[n][tags.size()];
				double[][] backward = new double[n][tags.size()];
				
				// @rhs
				if (useLogSpace) {
					Arrays.fill(forward[0], Double.NEGATIVE_INFINITY);
					Arrays.fill(backward[n-1], Double.NEGATIVE_INFINITY);
					forward[0][tags.get(START)] = 0; // @rhs: since it's in log space 
					backward[n-1][tags.get(END)] = 0; //
					fillValues(instance, forward, true, point, CRF::logSumOfExponentials, null);
					fillValues(instance, backward, false, point, CRF::logSumOfExponentials, null);
				} else {
					forward[0][tags.get(START)] = 1; 
					backward[n-1][tags.get(END)] = 1;
					fillValues(instance, forward, true, point, CRF::sum, null);
					fillValues(instance, backward, false, point, CRF::sum, null);
				}
				
				forwards.put(instance, forward);
				backwards.put(instance, backward);
//				System.out.println(instance);
//				System.out.println("Forward:");
//				for(int j=0; j<tags.size(); j++){
//					System.out.printf("%15s ", reverseTags[j]);
//				}
//				System.out.println();
//				for(int i=0; i<n; i++){
//					for(int j=0; j<tags.size(); j++){
//						System.out.printf("%15.2f ", forward[i][j]);
//					}
//					System.out.println();
//				}
//				System.out.println("Backward:");
//				for(int i=0; i<n; i++){
//					for(int j=0; j<tags.size(); j++){
//						System.out.printf("%15.2f ", backward[i][j]);
//					}
//					System.out.println();
//				}
//				total++;
//				if(total % 10000 == 0){
//					System.out.println(String.format("Completed %d/%d", total, size));
//				}
			}
		}
		
		private double[] computeGradient(double[] point){
			double[] result = new double[point.length];
//			System.out.println("Computing model distribution...");
			double[] modelDistribution = computeModelDistribution(point);
//			System.out.println("Computing regularization...");
			double[] regularization = computeRegularization(point);
			for(int i=0; i<result.length; i++){
				result[i] = empiricalDistribution[i] - modelDistribution[i] - regularization[i];
//				if (i == 36 || i == 0) {
//					System.out.println("Statistics for "+i);
//					System.out.println("Empirical "+empiricalDistribution[i]);
//					System.out.println("Model "+modelDistribution[i]);
//					System.out.println("Reg "+regularization[i]);
//					if (i == 36)
//					System.out.println("Predicted model dist: "+ (empiricalDistribution[i]-regularization[i]-0.094));
//				}
				//System.out.printf("%.3f ", result[i]);
			}
			//System.out.println();
			return result;
		}
		
		/**
		 * Compute model distribution
		 * sum_trainingdata 1/normalization (sum j=1 to n sum_tags sum_nextTag f_i(s,s',x,j)*forward_j(s|x)*factor(x,s,s')*backward_j+1(s'|x))
		 * @param point
		 * @return
		 */
		private double[] computeModelDistribution(double[] point){
			double[] result = new double[featureIndices.size()];
			double[] instanceExpectation = new double[featureIndices.size()];
			Arrays.fill(result, 0);
			for(Instance instance: trainingData){
				Arrays.fill(instanceExpectation, 0);
				int n = instance.words.size()+2;
				double[][] forward = forwards.get(instance);
				double[][] backward = backwards.get(instance);
				for(int j=0; j<n-1; j++){
					for(Tag curTag: tags.keySet()){
						int curTagIdx = tags.get(curTag);
						for(int nextTagIdx: getNextTags(curTag, j+1, n)){
							Tag nextTag = reverseTags[nextTagIdx];
							double factor = computeFactor(point, instance, j, curTag, nextTag, !useLogSpace); // @rhs: only take the power if we are in log space
							for(int i: featuresPresent(instance, j, curTag, nextTag)){
								if(i == -1) continue;
								if (useLogSpace) {
									instanceExpectation[i] += Math.exp(forward[j][curTagIdx]+factor+backward[j+1][nextTagIdx]-backward[0][tags.get(START)]);
								} else {
									instanceExpectation[i] += forward[j][curTagIdx]*factor*backward[j+1][nextTagIdx]/backward[0][tags.get(START)];
								}
							}
						}
					}
				}
				for(int i=0; i<featureIndices.size(); i++){
					//result[i] += instanceExpectation[i]/backward[0][tags.get(START)];
					result[i] += instanceExpectation[i];
				}
			}
			return result;
		}
		
		private double computeFactor(double[] point, Instance instance, int position, Tag prevTag, Tag curTag, boolean useExp){ //@rhs
			double result = 0;
			for(int i: featuresPresent(instance, position, prevTag, curTag)){
				if(i == -1) continue;
				result += point[i];
			}
			if (useExp) {
				return Math.exp(result);
			} else {
				return result;
			}
		}
		
		/**
		 * Compute lambda/sigma^2 for calculating regularization in gradient
		 * @param point
		 * @return
		 */
		private double[] computeRegularization(double[] point){
			double[] result = new double[featureIndices.size()];
			for(int i=0; i<result.length; i++){
				result[i] = point[i]/Math.pow(regularizationParameter, 2);
			}
			return result;
		}
		
		private double normalizationConstant(Instance instance){
			return backwards.get(instance)[0][tags.get(START)];
		}
	}
	
	private int[] getNextTags(Tag curTag, int position, int n){
		if(curTag == END){
			return empty;
		}
		if(position == n-2){
			return onlyEnd;
		}
		if(position == 0){
			if(curTag == START){
				return noStart;
			}
			return empty;
		}
		return noStart;
	}
	
	private int[] getPreviousTags(Tag curTag, int position, int n){
		if(curTag == START){
			return empty;
		}
		if(position == 1){
			return onlyStart;
		}
		if(position == n-1){
			if(curTag == END){
				return noEnd;
			}
			return empty;
		}
		return noEnd;
	}
	
	private void fillValues(Instance instance, double[][] lattice, boolean isForward, double[] weights, Function<double[], AccumulatorResult> accumulator, int[][] parentIdx){
		int n = instance.words.size()+2;
		int start, end, step;
		if(isForward){
			start = 1;
			end = n-1;
			step = 1;
		} else {
			start = n-2;
			end = 0;
			step = -1;
		}
		int tagSize = lattice[0].length;
		int instanceLength = lattice.length;
		double[] values = new double[tagSize];
		double value=0;
		for(int i=start; i*step<=end*step; i+=step){
			double[] prevValues = lattice[i-step];
			double[] curValues = lattice[i];
			int[] curParentIdx = (parentIdx != null) ? parentIdx[i] : null;
			for(int curTagIdx=0; curTagIdx<tagSize; curTagIdx++){
				Tag curTag = reverseTags[curTagIdx];
				Arrays.fill(values, Double.NaN);
				int[] reachableTags;
				if(isForward){
					reachableTags = getPreviousTags(curTag, i, instanceLength);
				} else {
					reachableTags = getNextTags(curTag, i, instanceLength);
				}
				for(int reachableTagIdx: reachableTags){
					Tag reachableTag = reverseTags[reachableTagIdx];
					Tag prevTagArg, curTagArg;
					int position;
					if(isForward){
						prevTagArg = reachableTag;
						curTagArg = curTag;
						position = i-1;
					} else {
						prevTagArg = curTag;
						curTagArg = reachableTag;
						position = i;
					}
					value = 0.0;
					for(int j: featuresPresent(instance, position, prevTagArg, curTagArg)){
						if(j == -1) continue;
						value += weights[j];
					}
//					if(!isForward){
//						System.out.printf("Pos: %d, Prev: %s, Cur: %s, Value: %.3f\n", position, prevTagArg, curTagArg, value);
//					}
					
					if (useLogSpace) {
						values[reachableTagIdx] = prevValues[reachableTagIdx]+value; // @rhs
					} else {
						values[reachableTagIdx] = prevValues[reachableTagIdx]*Math.exp(value);
					}
					
					//System.out.printf("%.3f ", values[reachableTagIdx]);
				}
				//System.out.println();
				AccumulatorResult result = accumulator.apply(values);
				curValues[curTagIdx] = result.value;
				if(curParentIdx != null){
					curParentIdx[curTagIdx] = result.maxIdx;
				}
			}
		}
	}
	
	/**
	 * First pass of the training data to get the number of words and tags
	 * @param trainingData
	 */
	private void readTrainingData(List<Instance> trainingData){
		for(Instance instance: trainingData){
			for(TaggedWord wordTag: instance.words){
				String word = wordTag.word();
				Tag tag = wordTag.tag();
				word = normalize(word, true);
				if(!words.containsKey(word)){
					words.put(word, words.size());
				}
				if(!tags.containsKey(tag)){
					tags.put(tag, tags.size());
				}
			}
		}
		tags.put(START, tags.size());
		tags.put(END, tags.size());
		words.put(UNKNOWN_WORD, words.size());
		
//		for(String feature: INIT_CAP_FEATURES){
//			words.put(feature, words.size());
//		}
//		for(String feature: END_FEATURES){
//			words.put(feature, words.size());
//		}
		
		reverseTags = new Tag[tags.size()];
		for(Tag tag: tags.keySet()){
			reverseTags[tags.get(tag)] = tag;
		}
		reverseWords = new String[words.size()];
		for(String word: words.keySet()){
			reverseWords[words.get(word)] = word;
		}
		
		noStart = new int[tags.size()-1];
		noEnd = new int[tags.size()-1];
		onlyStart = new int[]{tags.get(START)};
		onlyEnd = new int[]{tags.get(END)};
		empty = new int[0];
		int idx = 0;
		for(Tag tag: tags.keySet()){
			if(tag != START){
				noStart[idx] = tags.get(tag);
				idx++;
			}
		}
		idx = 0;
		for(Tag tag: tags.keySet()){
			if(tag != END){
				noEnd[idx] = tags.get(tag);
				idx++;
			}
		}
	}

	private void buildFeatures(List<Instance> trainingData) {
		featureIndices = new HashMap<String, Integer>();
		reverseFeatureIndices = new HashMap<Integer,String>();
		for(Instance instance: trainingData){
			List<TaggedWord> wordTags = instance.words;
			for(int position=0; position<wordTags.size(); position++){
				Tag prevTag = instance.getTagAt(position-1);
				Tag curTag = instance.getTagAt(position);
				insertFeatures(instance, position, prevTag, curTag);
			}
		}
		System.out.println("Num of featureIndices: "+featureIndices.size());
		System.out.println("Num of tags: "+tags.size());
	}
	
	public static class Feature{
		public String feature;
		public Tag prevTag;
		public Tag curTag;
		
		public Feature(String feature, Tag prevTag, Tag curTag){
			this.feature = feature;
			this.prevTag = prevTag;
			this.curTag = curTag;
		}
	}
	
	private void insertFeatures(Instance instance, int position, Tag prevTag, Tag curTag){
		String feature = null;
		for(int templateIdx=0; templateIdx<templates.length; templateIdx++){
			feature = templates[templateIdx].getFeature(instance, position, prevTag, curTag);
			if(!featureIndices.containsKey(feature)){
				featureIndices.put(feature, featureIndices.size());
				reverseFeatureIndices.put(featureIndices.size()-1, feature);
			}
//			for(Tag tag: tags.keySet()){
//				feature = feature.substring(0, feature.indexOf("|")+1) + tag.text + (feature.indexOf("0") >= 0 ? feature.substring(feature.indexOf("0")) : "");
//				if(!featureIndices.containsKey(feature)){
//					featureIndices.put(feature, featureIndices.size());
//				}
//				if(templates[templateIdx].isBigram){
//					for(Tag tag2: tags.keySet()){
//						feature = tag2.text + "|" + feature;
//						if(!featureIndices.containsKey(feature)){
//							featureIndices.put(feature, featureIndices.size());
//						}
//					}
//				}
//			}
		}
	}
	
	private int[] featuresPresent(Instance instance, int position, Tag prevTag, Tag curTag){
//		int[] result = new int[instance.featuresPresent.length];
//		int idx = 0;
//		for(Feature feature: instance.featuresPresent){
//			if(feature.prevTag == prevTag && feature.curTag == curTag){
//				result[idx] = featureIndices.get(feature.feature);
//			} else {
//				result[idx] = -1;
//			}
//			idx ++;
//		}
//		return result;
		int[] result = new int[templates.length];
		String feature = null;
		for(int templateIdx=0; templateIdx<templates.length; templateIdx++){
			feature = templates[templateIdx].getFeature(instance, position, prevTag, curTag);
			result[templateIdx] = featureIndices.getOrDefault(feature, -1);
		}
		return result;
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
		if(!isTraining && !words.containsKey(word)) return UNKNOWN_WORD;
		return word;
	}

	@Override
	public void train(List<Instance> trainingData) {
		System.out.println("Reading training data...");
		readTrainingData(trainingData);
		System.out.println("Building features...");
		buildFeatures(trainingData);
		System.out.println("Preparing for minimization...");
		LogLikelihood logLikelihood = new LogLikelihood(trainingData);
		double[] startingPoint = new double[featureIndices.size()];
		System.out.println("Starting point:");
		
		//startingPoint = new double[] {0.731, 1.977, 1.812, 1.611, 3.881, 0.264, 0.833, 2.491, 3.191, -0.465, 2.884, 3.800, 0.280, 2.739, 2.122, 1.157, 0.680, 3.384, 0.732, 1.915, 2.252, 0.983, 0.709, 0.849, 3.611, -0.393, 1.211, 0.001, 1.036, -0.129, 2.727, 1.305, 2.877, -0.195, -0.380, 2.072, 0.888, 2.093, 0.941, 0.472, -0.107, 2.529, 1.984, 2.333, 2.146, -1.077, 2.478, 0.225, 0.713, -0.285, 0.167, -0.523, 0.174, -0.187, 1.459, 0.742, 0.792, -0.411, -1.349, 0.802, 0.373, 0.321, 0.876, 0.074, 2.205, 0.976, 1.931, 0.316, -1.692, -0.278, 14.143, 12.332, 3.216, 0.699, -1.135, 1.014, 1.355, -0.598, -0.502, 1.545, -0.647, -1.290, -0.171, 1.005, 1.832, 1.236, -0.037, -1.467, 0.141, 0.755, 0.114, -0.216, 1.198, 1.327, 0.478, 0.500, 0.900, -0.671, 2.107, -0.447, 1.583, 1.434, 0.429, 2.418, 2.481, -0.903, -0.606, 2.107, -1.280, 1.110, 0.631, 0.466, -0.669, -0.476, -0.664, 1.029, -0.002, -0.138, -0.467, -0.387, 0.103, -0.494, 0.918, -0.503, 1.171, 1.557, -0.991, 1.143, -0.792, 0.811, -0.067, -0.239, -0.232, -0.322, 1.394, 1.338, -0.839, 0.798, -0.165, 0.129, 0.781, 0.569, 0.200, 1.148, 0.834, 1.406, -0.416, -1.554, -0.626, 1.067, -0.268, -0.252, 0.455, 0.098, -0.809, 0.877, 1.576, 1.436, 0.926, 0.456, -1.899, -0.457, 1.053, 0.917, 1.239, 3.382, 1.233, 1.075, 0.339, 1.274, 0.705, 0.458, 1.044, -0.711, 1.220, 1.596, 1.938, 1.218, 3.109, -0.519, 0.060, 0.866, -0.663, -0.262, -0.198, -0.153, -0.262, -0.933, -1.214, -0.170, -0.367, 0.318, -1.236, 0.562, 0.835, -0.490, 0.930, 0.433, -1.026, -1.046, 0.210, -0.058, -0.747, 0.658, -0.463, -1.341, -0.268, -0.385, -0.722, -0.232, 0.283, 0.338, 0.568, 0.774, -0.171, -2.054, 0.531, 0.339, 1.081, -0.930, 0.425, 0.381, 0.523, 0.197, -0.164, -0.669, -0.517, -0.273, 0.063, -0.207, -1.913, 0.470, 0.638, -0.185, 0.680, -0.242, 0.954, -0.332, 2.355, 1.109, 0.185, -1.484, -0.399, 0.121, 0.259, 0.650, 0.213, 0.868, 1.844, 0.728, -0.636, 1.374, 0.507, 0.270, -1.011, -1.249, -0.062, 0.630, 0.888, -1.279, -0.331, 0.110, -0.773, 1.338, 0.513, -1.172, -0.298, 1.787, -1.027, 1.669, -0.174, 2.723, 1.729, 0.774, 1.652, -0.027, -0.580, 1.464, 0.006, 0.235, 0.440, 0.367, 0.646, 1.367, 0.472, 0.893, 0.420, 1.003, 2.055, -0.797, -1.265, 1.201, -0.197, 0.842, 0.177, 0.152, 0.955, 1.250, 0.265, -1.105, 0.226, -0.586, 0.302, 0.287, -0.697, -0.246, 0.078, 0.235, -1.133, -1.495, 0.140, 0.789, 0.378, 0.338, 1.308, -0.205, -0.362, 0.538, 0.764, 0.808, -0.115, 0.106, 0.802, -0.013, -1.297, 0.644, 1.079, -0.713, 0.602, 0.289, -0.718, -0.359, -0.674, 0.044, 0.483, -0.572, 0.487, 0.069, 0.918, -1.115, -1.095, 0.013, -0.353, 0.596, 0.366, 0.049, 0.766, -0.061, -0.730, 0.783, -0.076, 0.034, 0.275, 1.791, -0.039, -0.384, -0.915, 0.574, 1.377, 0.053, 0.537, -0.350, -0.291, 0.341, -0.864, 0.202, -0.513, 0.643, -1.402, -0.510, -0.504, -0.183, 1.219, -0.852, 1.857, -0.527, 0.218, -1.359, -0.269, -1.758, 0.177, -0.459, -0.356, 0.645, 0.395, 0.804, 0.342, 0.751, 1.411, 1.308, -0.451, -1.033, 0.531, 0.189, -0.255, 0.782, 0.259, -0.189, -1.557, -0.118, -0.179, 0.451, 0.566, 0.078, 0.343, 0.381, -1.210, 0.374, -1.067, -0.414, -0.061, -0.200, -1.358, -1.285, 0.063, -0.807, 0.470, -0.372, -0.970, 0.326, -1.839, 1.008, -0.592, 0.347, -0.040, 0.078, -1.855, 0.736, -0.336, 1.259, 0.406, 2.263, -0.043, 0.358, -1.542, 0.025, 0.732, 0.210, 0.789, 0.311, 0.625, 1.199, -0.935, 1.142, 0.594, 0.332, -1.258, 0.675, -0.789, 1.237, 1.076, 0.451, 1.088, -0.365, -0.352, 0.257, -0.005, -0.099, 0.824, 0.063, 0.515, 0.663, -0.379, -0.172, -0.289, -0.150, 0.852, 0.876, -1.322, 0.961, -0.528, -1.228, 0.409, -0.027, -0.961, 1.577, -0.392, 0.695, -0.339, 0.714, 0.795, -1.244, 0.598, -0.852, -1.186, 0.423, 0.304, -0.212, -0.670, 0.709, -0.522, 0.770, 0.692, -0.166, -2.317, 0.151, 0.903, -0.606, -1.005, 0.640, -0.622, -0.373, -0.289, -0.127, -0.268, -0.776, -0.720, 1.025, 0.309, 1.551, -0.169, -0.436, 0.107, 0.450, 1.522, 0.802, 0.571, 1.136, 0.523, 1.209, -0.026, 0.425, 0.708, 0.099, -0.766, -0.169, -0.073, -0.513, -0.611, 0.728, 0.904, -0.392, 0.719, -0.418, 0.117, 0.273, -0.254, 0.327, 0.076, 1.612, -0.019, -1.031, -0.032, -0.023, 0.015, 0.248, -1.250, -0.602, -1.742, 0.732, -0.631, 0.007, 0.239, 0.238, -0.686, 1.334, 1.327, 0.708, -0.246, 0.872, 1.644, -0.975, -0.676, 0.192, 1.620, 0.645, 0.715, 1.586, -1.270, -0.269, 0.242, 0.821, -0.338, 0.389, 0.826, 0.190, 0.791, -1.653, 0.281, 0.158, 0.109, -0.892, 0.099, 0.987, 0.013, -1.082, -0.912, -0.401, 0.162, 0.739, 0.825, 0.464, -0.285, 0.040, 0.605, 0.379, 0.215, -1.769, -0.667, 0.125, 0.369, 0.391, 0.531, 0.631, 0.510, 1.394, 0.301, -1.104, -0.037, -0.303, -0.692, 0.585, 1.175, -0.653, 0.422, -0.134, -1.112, 1.172, -0.449, 0.861, -0.657, -0.286, 1.102, 1.176, 0.849, 0.862, 1.314, 0.332, -0.093, 1.095, 1.162, 0.123, 1.423, 0.737, -0.162, -1.015, -0.062, 0.298, 0.529, -0.045, -0.965, 0.248, 0.792, -0.050, -1.416, 0.832, -0.182, 0.553, -0.428, 0.221, 0.966, 0.842, -1.071, 1.490, 0.540, 0.986, 0.583, -0.823, 2.027, 1.256, -0.657, -0.548, 0.364, -0.049, -0.724, 0.037, -0.748, 0.472, 0.943, 0.281, -0.064, -0.641, 0.930, -1.087, 0.253, 1.422, -1.318, -0.183, -1.223, -0.345, -0.083, 0.197, -0.183, 0.875, 0.946, -0.341, 0.037, -1.687, -0.352, 1.154, 1.319, 0.421, 0.803, -0.279, 0.507, 0.414, 0.647, 0.095, 1.283, 0.022, 1.406, -0.644, -0.064, 0.503, 0.025, 1.071, -0.936, -1.263, -0.285, -0.083, -0.604, 0.305, 0.845, -0.590, -0.516, 0.484, 0.918, -0.955, 0.532, -1.644, -1.250, -1.153, -0.361, -0.332, 0.297, 0.121, 0.278, -0.149, -0.659, 0.648, 0.159, -0.388, -0.068, -1.162, -0.516, -0.090, 0.464, -0.881, 1.703, 0.180, 0.326, -0.588, 1.524, 0.084, -1.073, 0.131, 0.075, 0.065, 1.663, -0.452, -0.099, -0.500, 0.402, 0.944, -0.146, -1.166, 0.998, 0.093, -0.232, 0.844, -0.789, -1.174, 0.148, 1.202, -0.454, -0.516, 1.117, 0.329, 0.607, 0.827, 0.222, 0.804, 0.050, -1.246, -0.171, -1.248, 0.346, 0.093, -0.350, 0.758, -1.035, -0.073, -1.688, -1.796, -0.078, -0.090, 0.008, -1.156, 0.038, 0.681, -1.597, 0.463, 1.262, 1.345, -0.341, 0.469, 0.446, 1.234, -0.392, -0.986, -0.142, 1.003, -1.346, -0.474, -1.126, -1.293, 0.059, 0.564, -0.101, -1.188, 0.032, -0.842, -1.873, 0.295, 0.692, -0.492, 0.066, -0.103, 0.198, 0.101, 0.727, 0.193, -0.303, 0.960, 0.356, 0.331, 0.055, 1.221, -0.067, -0.031, -0.395, 0.806, -0.223, 1.207, 1.187, 0.005, -0.054, -0.239, 1.634, 1.865, -0.639, -0.161, -0.125, 0.003, -0.315, -0.315, 0.009, -0.833, 0.007, -0.446, 0.729, 0.675, 0.832, -0.684, -0.286, 0.013, 0.039, 0.911, -1.513, -1.454, 0.473, -0.289, 0.215, 0.815, 0.998, -0.544, -0.179, 0.176, -0.611, 1.351, 0.380, 1.120, 0.800, 0.256, 1.280, -1.438, 0.303, -0.998, -0.186, 0.205, -0.500, -0.560, -0.545, 0.798, -0.234, 2.114, 0.518, 1.755, -0.059, -0.908, 0.550, 0.097, 0.979, 0.527, -0.374, 1.001, 0.146, 0.222, -0.458, 0.232, 0.907, -0.457, 0.603, 0.314, -0.209, 0.672, 1.870, -0.762, 0.599, 0.125, 1.723, 0.476, -1.052, -0.809, -0.351, 0.775, 0.071, -0.163, 0.653, 1.452, 0.938, -0.010, 1.248, -0.248, 0.468, 0.518, 0.217, 0.975, -0.746, 0.394, -0.002, -1.676, -0.920, 0.765, -0.451, 1.249, -0.843, 1.125, 0.136, -1.556, -0.177, 0.868, 1.594, 0.021, -1.186, 1.501, -1.224, 0.815, -0.021, -0.118, 0.664, 0.059, 0.051, 1.304, -0.832, -0.731, -0.260, 0.354, 0.766, -0.786, 0.106, -1.008, 0.921, -0.270, 1.014, 0.993, 0.624, -1.482, 1.443, 0.834, -0.974, -0.547, 0.545, -0.026, -0.226, 1.010, -0.101, -0.003, 0.405, 0.060, -0.865, 0.847, 0.010, 0.383, 0.381, 0.535, -1.179, 0.362, 0.186, -0.249, 0.451, 0.608, 0.324, 0.575, 0.230, -0.576, 0.344, 1.048, -0.362, -0.073, 0.635, 0.126, 1.553, -1.761, -0.386, 0.689, 0.276, 1.032, 0.044, -0.258, 0.646, -0.657, 0.065, 0.205, -1.146, 0.624, -0.696, 0.881, -0.862, 0.723, 0.677, 0.428, 0.203, 0.036, 0.165};
		//startingPoint = new double[] {0.322 ,2.250 ,1.237 ,2.052 ,4.267 ,1.992 ,0.786 ,1.852 ,3.115 ,-0.010 ,2.539 ,5.836 ,0.986 ,3.952 ,3.854 ,0.430 ,0.706 ,4.603 ,0.035 ,2.320 ,3.343 ,1.111 ,1.066 ,0.243 ,4.081 ,0.196 ,0.773 ,0.182 ,0.508 ,0.063 ,4.834 ,1.843 ,3.562 ,1.293 ,-0.112 ,1.649 ,0.443 ,1.882 ,1.826 ,1.823 ,0.097 ,2.201 ,1.929 ,1.986 ,7.007 ,0.177 ,3.519 ,0.190 ,0.900 ,-0.005 ,0.855 ,0.608 ,0.461 ,2.084 ,0.656 ,0.247 ,0.950 ,0.107 ,0.209 ,1.274 ,0.118 ,0.436 ,0.485 ,0.133 ,1.575 ,2.112 ,2.297 ,1.323 ,-0.075 ,0.046 ,62.523 ,62.226 ,7.286 ,0.940 ,-0.774 ,2.821 ,1.650 ,-1.722 ,-0.009 ,0.841 ,0.362 ,0.146 ,-1.172 ,0.387 ,2.391 ,2.294 ,-0.693 ,-0.004 ,0.264 ,0.627 ,0.107 ,0.020 ,0.215 ,1.427 ,-0.084 ,0.124 ,0.349 ,1.024 ,2.390 ,0.190 ,2.449 ,-0.227 ,0.243 ,4.106 ,4.617 ,0.448 ,-0.799 ,1.851 ,0.152 ,0.242 ,0.238 ,1.149 ,-0.386 ,0.052 ,-0.029 ,1.019 ,0.851 ,0.091 ,0.304 ,0.084 ,0.113 ,0.117 ,1.204 ,0.042 ,0.366 ,0.334 ,0.742 ,1.162 ,0.127 ,0.316 ,0.120 ,-0.008 ,0.116 ,0.174 ,2.069 ,0.489 ,0.301 ,0.229 ,0.173 ,0.300 ,0.905 ,0.630 ,0.492 ,0.488 ,0.235 ,1.034 ,0.110 ,-0.107 ,-0.029 ,0.256 ,0.143 ,0.150 ,0.218 ,-0.152 ,0.497 ,3.489 ,1.980 ,2.253 ,0.447 ,0.134 ,0.007 ,0.032 ,0.222 ,0.883 ,0.201 ,3.865 ,1.468 ,0.422 ,2.539 ,3.809 ,0.767 ,0.339 ,0.203 ,-0.067 ,3.638 ,3.774 ,3.372 ,1.060 ,4.239 ,0.011 ,0.138 ,0.476 ,0.008 ,0.157 ,1.002 ,0.022 ,0.005 ,-0.031 ,0.515 ,-1.116 ,0.581 ,1.016 ,-0.153 ,0.596 ,2.141 ,0.550 ,0.787 ,0.158 ,-0.033 ,0.122 ,0.591 ,0.104 ,-0.056 ,0.113 ,-0.018 ,-0.194 ,-0.009 ,-0.017 ,-0.050 ,0.126 ,0.192 ,0.283 ,4.061 ,0.488 ,0.226 ,-0.295 ,0.359 ,0.281 ,0.650 ,-2.064 ,0.634 ,0.144 ,0.145 ,1.429 ,0.513 ,0.430 ,0.658 ,0.186 ,0.680 ,0.609 ,0.330 ,0.805 ,0.832 ,0.723 ,4.090 ,0.507 ,0.703 ,0.372 ,0.510 ,0.275 ,0.602 ,0.049 ,-0.268 ,0.431 ,0.453 ,0.243 ,0.126 ,0.259 ,0.595 ,1.791 ,-0.005 ,0.197 ,0.833 ,0.665 ,0.034 ,-0.170 ,-0.116 ,0.146 ,0.161 ,0.162 ,-0.006 ,0.102 ,-0.042 ,0.558 ,0.274 ,-0.641 ,0.031 ,3.493 ,0.137 ,0.312 ,0.066 ,0.535 ,1.695 ,0.406 ,2.172 ,0.202 ,0.303 ,0.303 ,0.334 ,0.085 ,0.105 ,0.391 ,0.434 ,1.278 ,1.610 ,2.517 ,0.604 ,1.884 ,3.610 ,-0.282 ,0.414 ,0.212 ,0.101 ,0.879 ,1.451 ,0.520 ,0.618 ,0.639 ,-0.368 ,0.153 ,0.183 ,-0.027 ,0.075 ,0.116 ,0.500 ,0.134 ,0.362 ,-0.628 ,0.046 ,0.112 ,0.736 ,0.610 ,0.261 ,0.851 ,0.228 ,0.000 ,-0.025 ,0.502 ,-0.222 ,0.102 ,0.068 ,0.160 ,0.802 ,0.887 ,0.102 ,0.852 ,2.153 ,0.922 ,1.290 ,1.238 ,0.544 ,-0.062 ,-0.024 ,0.335 ,0.220 ,-0.025 ,0.470 ,0.420 ,0.578 ,-2.102 ,0.137 ,1.103 ,1.043 ,0.911 ,1.798 ,0.110 ,1.219 ,0.085 ,0.681 ,0.682 ,0.874 ,0.829 ,0.712 ,1.049 ,0.463 ,0.132 ,0.078 ,0.500 ,1.024 ,0.138 ,1.286 ,0.619 ,0.678 ,0.128 ,0.062 ,0.299 ,-0.391 ,0.252 ,-0.195 ,0.328 ,0.547 ,1.157 ,0.716 ,0.174 ,0.529 ,0.289 ,-0.521 ,-0.111 ,0.136 ,-0.050 ,0.086 ,-0.032 ,0.236 ,0.333 ,0.156 ,0.912 ,0.626 ,1.747 ,0.590 ,0.286 ,-0.088 ,0.054 ,0.341 ,1.558 ,-0.112 ,0.146 ,-0.398 ,0.074 ,0.148 ,0.759 ,0.482 ,0.612 ,-0.710 ,0.091 ,0.421 ,0.490 ,-0.242 ,0.136 ,0.313 ,-0.084 ,0.234 ,0.431 ,-0.138 ,-0.043 ,1.110 ,-0.088 ,0.145 ,0.172 ,1.049 ,1.025 ,0.512 ,1.706 ,0.085 ,0.499 ,0.568 ,0.269 ,0.023 ,0.167 ,0.167 ,0.297 ,0.443 ,2.407 ,0.117 ,0.932 ,0.126 ,0.540 ,2.220 ,0.177 ,0.156 ,0.189 ,0.157 ,0.150 ,0.006 ,0.741 ,0.899 ,0.827 ,0.620 ,0.581 ,-0.660 ,0.295 ,0.267 ,0.884 ,0.434 ,0.744 ,1.280 ,1.493 ,0.963 ,0.122 ,0.438 ,0.293 ,0.516 ,0.891 ,-0.037 ,0.032 ,0.234 ,0.119 ,0.688 ,0.796 ,1.068 ,0.941 ,0.020 ,0.269 ,0.154 ,1.082 ,0.980 ,1.979 ,0.581 ,-0.146 ,0.125 ,0.186 ,0.392 ,-0.012 ,0.895 ,0.410 ,-0.244 ,0.170 ,0.113 ,0.255 ,0.053 ,0.038 ,0.002 ,0.041 ,0.245 ,-0.160 ,-0.420 ,0.055 ,0.207 ,0.067 ,0.048 ,0.898 ,0.583 ,0.769 ,0.962 ,0.024 ,0.577 ,1.520 ,-0.369 ,0.274 ,0.320 ,0.376 ,0.717 ,0.458 ,0.165 ,0.427 ,0.848 ,0.582 ,0.131 ,0.458 ,-0.008 ,-0.076 ,0.818 ,0.856 ,0.169 ,0.470 ,0.378 ,0.076 ,0.009 ,0.057 ,0.095 ,0.253 ,0.158 ,-0.055 ,0.186 ,0.411 ,0.415 ,0.093 ,0.445 ,1.999 ,0.090 ,0.317 ,0.106 ,0.048 ,0.094 ,0.528 ,-0.023 ,0.799 ,-0.484 ,0.223 ,-0.245 ,0.552 ,0.050 ,0.837 ,0.097 ,0.255 ,0.229 ,0.239 ,0.402 ,0.250 ,0.682 ,0.211 ,0.348 ,0.128 ,-0.034 ,0.089 ,0.336 ,0.428 ,1.660 ,0.343 ,0.264 ,0.130 ,0.073 ,1.519 ,1.329 ,-0.195 ,0.222 ,0.155 ,0.247 ,-0.083 ,0.179 ,0.594 ,0.589 ,-0.091 ,0.146 ,0.250 ,-0.609 ,-0.012 ,-0.099 ,0.041 ,0.386 ,0.168 ,0.430 ,0.147 ,-0.012 ,0.288 ,0.154 ,0.128 ,0.353 ,-0.239 ,0.027 ,0.060 ,0.108 ,0.229 ,0.017 ,0.297 ,0.110 ,0.401 ,0.124 ,-0.094 ,0.106 ,-0.120 ,0.301 ,0.679 ,0.504 ,-0.008 ,0.126 ,0.031 ,-0.200 ,0.967 ,0.071 ,0.099 ,-0.003 ,0.014 ,0.212 ,0.261 ,0.166 ,0.585 ,0.871 ,-0.113 ,0.014 ,0.274 ,0.379 ,0.089 ,0.340 ,0.151 ,0.305 ,-0.915 ,0.243 ,0.076 ,0.196 ,0.225 ,-0.054 ,0.729 ,0.147 ,0.038 ,0.090 ,0.902 ,0.253 ,0.393 ,0.083 ,0.153 ,0.206 ,0.210 ,-0.020 ,0.574 ,0.195 ,0.385 ,0.298 ,-0.043 ,0.169 ,0.287 ,0.000 ,-0.065 ,0.087 ,0.736 ,-0.088 ,0.022 ,-0.050 ,0.749 ,0.816 ,0.077 ,0.378 ,-0.031 ,0.170 ,-0.112 ,0.069 ,0.597 ,0.155 ,0.548 ,0.267 ,-0.011 ,0.055 ,-0.038 ,0.067 ,0.196 ,0.289 ,0.078 ,0.258 ,-0.114 ,0.043 ,0.228 ,1.718 ,0.191 ,0.610 ,0.210 ,0.399 ,0.865 ,0.142 ,0.139 ,0.242 ,0.084 ,0.246 ,-0.016 ,0.021 ,0.209 ,0.131 ,0.527 ,0.202 ,-0.154 ,-0.017 ,0.083 ,-0.036 ,0.176 ,0.211 ,0.533 ,0.161 ,0.384 ,0.288 ,0.655 ,0.440 ,0.196 ,-0.048 ,-0.052 ,-0.596 ,-0.591 ,0.321 ,0.134 ,0.209 ,0.061 ,0.300 ,0.195 ,0.831 ,0.231 ,0.224 ,0.042 ,-0.021 ,0.215 ,0.091 ,0.905 ,0.363 ,0.146 ,0.160 ,-0.010 ,0.317 ,0.065 ,-0.068 ,0.146 ,0.066 ,0.125 ,0.544 ,0.130 ,0.188 ,-0.051 ,-0.073 ,0.960 ,0.075 ,0.235 ,0.318 ,0.223 ,0.736 ,1.384 ,1.160 ,0.631 ,0.391 ,0.568 ,0.224 ,0.103 ,0.697 ,0.157 ,0.504 ,0.357 ,0.277 ,0.398 ,0.324 ,-0.028 ,0.342 ,0.287 ,0.066 ,0.427 ,0.448 ,0.445 ,0.047 ,0.480 ,0.383 ,0.199 ,1.145 ,0.389 ,0.166 ,0.098 ,0.212 ,-0.082 ,0.952 ,0.410 ,-1.008 ,0.935 ,0.560 ,0.842 ,0.727 ,1.574 ,0.027 ,-0.102 ,0.620 ,0.241 ,-0.118 ,-0.016 ,-0.113 ,-0.079 ,0.088 ,0.100 ,0.225 ,-0.034 ,0.457 ,0.475 ,0.480 ,0.858 ,0.551 ,0.657 ,0.458 ,0.074 ,0.714 ,0.208 ,0.136 ,0.072 ,0.004 ,0.063 ,0.312 ,0.108 ,0.734 ,0.918 ,0.697 ,0.678 ,-0.541 ,0.201 ,-0.002 ,0.634 ,0.090 ,0.053 ,0.035 ,0.019 ,0.746 ,0.519 ,0.270 ,0.041 ,0.065 ,0.122 ,0.255 ,0.298 ,0.240 ,-0.977 ,0.443 ,0.339 ,0.440 ,0.173 ,0.384 ,0.340 ,-0.106 ,0.371 ,2.011 ,0.943 ,-0.128 ,-0.181 ,0.206 ,-1.027 ,-0.157 ,0.117 ,0.311 ,-0.173 ,0.231 ,0.148 ,0.073 ,0.725 ,0.055 ,0.293 ,0.564 ,0.126 ,0.364 ,-0.111 ,0.105 ,-0.018 ,0.034 ,0.086 ,-0.034 ,0.109 ,-0.256 ,0.233 ,0.114 ,0.266 ,0.176 ,0.326 ,0.073 ,-0.383 ,0.737 ,0.813 ,0.432 ,0.412 ,0.128 ,0.480 ,0.454 ,0.116 ,0.111 ,0.088 ,0.707 ,0.185 ,0.354 ,0.183 ,0.415 ,0.156 ,0.341 ,-0.080 ,0.363 ,0.057 ,0.318 ,0.196 ,-0.126 ,0.337 ,0.037 ,0.159 ,0.611 ,0.640 ,0.194 ,0.420 ,-0.376 ,0.425 ,0.500 ,0.054 ,0.551 ,1.420 ,0.811 ,0.233 ,0.470 ,0.167 ,0.046 ,-0.181 ,-0.034 ,0.190 ,0.314 ,0.298 ,-0.100 ,0.787 ,0.569 ,-0.207 ,0.038 ,-0.222 ,0.275 ,0.036 ,-0.137 ,0.284 ,-0.027 ,0.429 ,0.400 ,0.107 ,0.176 ,0.168 ,0.569 ,0.775 ,-0.083 ,-0.003 ,-0.010 ,0.078 ,0.341 ,0.176 ,0.254 ,0.252 ,0.435 ,0.248 ,0.971 ,0.187 ,0.223 ,-0.191 ,0.324 ,0.519 ,-0.063 ,-0.067 ,0.493 ,0.149 ,0.337 ,0.476 ,0.222 ,0.084 ,0.204 ,0.333 ,-0.969 ,0.206 ,0.034 ,0.545 ,0.105 ,0.144 ,-0.080 ,0.125 ,0.082 ,-0.032 ,0.101 ,0.163 ,0.442 ,0.140 ,0.142 ,-0.003 ,-0.128 ,0.320 ,0.219 ,-0.067 ,0.196 ,0.175 ,0.542 ,0.030 ,0.060 ,0.430 ,0.006 ,0.256 ,0.112 ,0.050 ,0.172 ,-0.028 ,0.399 ,0.592 ,-0.117 ,0.137 ,0.623 ,0.209 ,-0.055 ,0.839 ,0.294 ,0.483 ,0.157 ,0.149 ,0.182};
//		startingPoint = new double[] {0.465, 2.255, 1.427, 1.972, 4.146, 1.398, 0.739, 2.021, 3.162, -0.162, 2.741, 5.243, 0.876, 3.745, 3.324, 0.609, 0.688, 4.259, 0.169, 2.192, 3.031, 1.099, 0.976, 0.431, 4.036, 0.080, 0.898, 0.134, 0.738, -0.002, 4.314, 1.773, 3.376, 0.861, -0.246, 1.745, 0.558, 2.113, 1.731, 1.372, 0.041, 2.403, 1.950, 2.106, 5.849, -0.190, 3.429, 0.291, 0.835, -0.089, 0.623, 0.250, 0.409, 1.824, 0.788, 0.389, 0.982, -0.037, -0.235, 1.189, 0.203, 0.587, 0.656, 0.023, 1.885, 1.880, 2.333, 1.064, -0.552, -0.058, 46.108, 45.297, 6.250, 0.859, -0.743, 2.381, 1.608, -1.501, -0.161, 1.043, 0.096, -0.303, -0.959, 0.590, 2.338, 2.072, -0.487, -0.410, 0.225, 0.683, 0.114, -0.059, 0.544, 1.437, 0.055, 0.251, 0.523, 0.659, 2.491, 0.075, 2.412, 0.165, 0.303, 3.822, 3.966, 0.057, -0.735, 1.939, -0.258, 0.527, 0.373, 1.005, -0.527, -0.122, -0.233, 1.069, 0.608, 0.013, 0.118, -0.048, 0.108, -0.058, 1.218, -0.113, 0.621, 0.725, 0.379, 1.265, -0.134, 0.471, 0.059, -0.087, -0.001, 0.013, 1.944, 0.773, 0.051, 0.384, 0.088, 0.254, 0.970, 0.611, 0.450, 0.675, 0.426, 1.227, -0.044, -0.569, -0.265, 0.390, -0.029, 0.032, 0.272, -0.167, 0.142, 2.692, 2.015, 2.142, 0.574, 0.234, -0.506, -0.093, 0.494, 0.950, 0.514, 3.710, 1.312, 0.679, 2.187, 3.410, 0.738, 0.306, 0.481, -0.283, 2.948, 3.183, 2.976, 1.109, 3.979, -0.158, 0.110, 0.609, -0.128, 0.033, 0.849, -0.032, -0.078, -0.313, -0.038, -0.889, 0.424, 0.967, -0.510, 0.588, 1.906, 0.267, 0.798, 0.249, -0.346, -0.179, 0.490, 0.062, -0.283, 0.223, -0.160, -0.577, -0.098, -0.134, -0.269, 0.050, 0.225, 0.332, 3.266, 0.581, 0.088, -0.811, 0.459, 0.292, 0.798, -1.871, 0.582, 0.216, 0.352, 0.918, 0.351, 0.125, 0.280, 0.268, 0.604, 0.402, -0.362, 0.686, 0.761, 0.432, 3.117, 0.283, 0.819, 0.182, 1.092, 0.545, 0.439, -0.358, -0.358, 0.446, 0.508, 0.375, 0.130, 0.445, 0.983, 1.650, -0.202, 0.507, 0.691, 0.422, -0.275, -0.528, -0.135, 0.305, 0.405, -0.261, -0.105, 0.127, -0.268, 0.783, 0.336, -0.829, -0.073, 3.336, -0.217, 0.763, 0.009, 1.249, 1.687, 0.540, 2.148, 0.153, 0.107, 0.687, 0.201, 0.132, 0.215, 0.412, 0.499, 1.394, 1.478, 2.294, 0.628, 1.694, 3.243, -0.446, 0.056, 0.544, 0.043, 0.869, 1.145, 0.376, 0.651, 0.817, -0.274, -0.196, 0.190, -0.213, 0.158, 0.170, 0.288, 0.022, 0.332, -0.567, -0.294, -0.344, 0.813, 0.636, 0.306, 0.695, 0.591, -0.066, -0.138, 0.500, 0.001, 0.317, 0.007, 0.140, 0.842, 0.739, -0.205, 0.874, 2.051, 0.484, 1.203, 1.062, 0.244, -0.026, -0.232, 0.271, 0.307, -0.201, 0.512, 0.298, 0.695, -1.947, -0.211, 0.873, 0.709, 0.772, 1.246, 0.097, 0.995, 0.045, 0.212, 0.660, 0.611, 0.560, 0.545, 1.254, 0.287, -0.013, -0.214, 0.527, 1.156, 0.095, 1.122, 0.310, 0.225, 0.199, -0.188, 0.262, -0.459, 0.359, -0.596, 0.058, 0.230, 0.864, 0.910, -0.108, 0.935, 0.004, -0.312, -0.496, 0.009, -0.568, 0.118, -0.174, 0.080, 0.409, 0.229, 0.867, 0.533, 1.386, 0.853, 0.622, -0.191, -0.263, 0.427, 1.468, -0.239, 0.358, -0.349, -0.012, -0.283, 0.478, 0.293, 0.704, -0.296, 0.072, 0.393, 0.467, -0.531, 0.254, -0.054, -0.203, 0.140, 0.338, -0.527, -0.407, 0.848, -0.319, 0.251, 0.019, 0.511, 0.923, -0.275, 1.432, -0.074, 0.440, 0.392, 0.247, -0.531, 0.354, 0.005, 0.606, 0.441, 2.426, 0.069, 0.947, -0.326, 0.392, 1.745, 0.189, 0.368, 0.234, 0.310, 0.418, -0.271, 0.883, 0.798, 0.552, -0.014, 0.601, -0.731, 0.600, 0.534, 0.734, 0.515, 0.381, 0.802, 1.084, 0.757, 0.045, 0.537, 0.253, 0.506, 0.890, -0.152, -0.039, 0.099, 0.033, 0.768, 0.823, 0.283, 0.885, -0.135, -0.186, 0.102, 0.844, 0.467, 1.865, 0.262, 0.001, -0.042, 0.353, 0.530, -0.406, 0.803, 0.018, -0.567, 0.251, 0.179, 0.140, -0.176, 0.202, -0.174, 0.207, 0.405, -0.179, -1.002, 0.087, 0.446, -0.115, -0.256, 0.816, 0.318, 0.384, 0.602, -0.033, 0.324, 0.833, -0.432, 0.509, 0.351, 0.758, 0.427, 0.217, 0.063, 0.408, 1.058, 0.660, 0.279, 0.659, 0.077, 0.179, 0.537, 0.726, 0.348, 0.400, 0.049, 0.007, -0.020, -0.110, -0.077, 0.404, 0.411, -0.169, 0.360, 0.144, 0.343, 0.151, 0.219, 1.658, 0.102, 0.746, 0.069, -0.283, 0.063, 0.361, -0.020, 0.631, -0.693, 0.035, -0.742, 0.655, -0.140, 0.555, 0.141, 0.270, -0.013, 0.608, 0.684, 0.406, 0.395, 0.430, 0.766, -0.173, -0.241, 0.118, 0.754, 0.527, 1.428, 0.745, -0.133, 0.103, 0.132, 1.363, 0.844, -0.004, 0.419, 0.168, 0.428, -0.561, 0.210, 0.463, 0.453, -0.358, 0.135, 0.495, -0.671, -0.364, -0.312, -0.092, 0.338, 0.355, 0.558, 0.253, -0.107, 0.212, 0.297, 0.207, 0.327, -0.746, -0.180, 0.084, 0.193, 0.290, 0.169, 0.398, 0.242, 0.721, 0.193, -0.406, 0.079, -0.197, 0.012, 0.661, 0.737, -0.203, 0.223, -0.018, -0.496, 1.029, -0.087, 0.298, -0.199, -0.095, 0.508, 0.564, 0.394, 0.692, 0.962, 0.042, -0.024, 0.539, 0.630, 0.092, 0.683, 0.341, 0.176, -0.935, 0.149, 0.149, 0.307, 0.149, -0.348, 0.633, 0.364, 0.010, -0.337, 0.872, 0.137, 0.455, -0.069, 0.175, 0.457, 0.420, -0.283, 0.845, 0.308, 0.560, 0.389, -0.299, 0.689, 0.604, -0.214, -0.225, 0.179, 0.495, -0.302, 0.026, -0.278, 0.704, 0.901, 0.147, 0.251, -0.235, 0.426, -0.432, 0.129, 0.834, -0.308, 0.361, -0.176, -0.121, 0.007, -0.053, -0.017, 0.421, 0.565, -0.011, 0.210, -0.601, -0.083, 0.533, 1.685, 0.268, 0.658, 0.060, 0.437, 0.723, 0.308, 0.127, 0.589, 0.059, 0.645, -0.220, -0.006, 0.391, 0.177, 0.710, -0.122, -0.515, -0.106, 0.027, -0.220, 0.218, 0.425, 0.179, -0.042, 0.418, 0.482, 0.116, 0.424, -0.356, -0.392, -0.387, -0.549, -0.536, 0.318, 0.134, 0.238, -0.013, 0.092, 0.333, 0.610, 0.061, 0.150, -0.316, -0.185, 0.138, 0.215, 0.386, 0.806, 0.160, 0.211, -0.200, 0.718, 0.068, -0.410, 0.157, 0.076, 0.113, 0.882, -0.014, 0.144, -0.203, 0.047, 0.960, 0.038, -0.155, 0.526, 0.192, 0.416, 1.226, 0.561, 0.017, 0.315, 0.760, 0.033, -0.076, 0.879, 0.216, 0.540, 0.510, 0.251, 0.537, 0.239, -0.455, 0.191, -0.164, 0.159, 0.425, 0.266, 0.539, -0.230, 0.384, -0.252, -0.377, 0.992, 0.245, 0.170, -0.108, 0.155, 0.198, 0.255, 0.514, -0.772, 1.081, 0.272, 0.742, 0.639, 1.462, -0.085, -0.382, 0.345, 0.492, -0.513, -0.163, -0.437, -0.457, 0.081, 0.256, 0.132, -0.392, 0.394, 0.084, -0.303, 0.674, 0.604, 0.276, 0.342, 0.024, 0.572, 0.126, 0.333, 0.112, -0.098, 0.283, 0.336, 0.182, 0.524, 1.045, 0.476, 0.474, -0.533, 0.399, -0.079, 0.788, 0.404, 0.042, 0.009, -0.067, 1.009, 0.940, 0.006, -0.019, 0.008, 0.085, 0.108, 0.134, 0.175, -0.989, 0.318, 0.098, 0.543, 0.357, 0.505, 0.090, -0.263, 0.249, 1.277, 0.934, -0.550, -0.577, 0.320, -1.022, -0.084, 0.350, 0.624, -0.279, 0.122, 0.174, -0.147, 0.903, 0.072, 0.556, 0.654, 0.161, 0.650, -0.526, 0.173, -0.289, -0.039, 0.129, -0.176, -0.086, -0.334, 0.420, -0.026, 0.682, 0.285, 0.802, 0.025, -0.558, 0.658, 0.577, 0.593, 0.473, -0.036, 0.648, 0.345, 0.156, -0.056, 0.133, 0.793, -0.001, 0.441, 0.229, 0.246, 0.324, 0.845, -0.309, 0.427, 0.083, 0.786, 0.297, -0.435, 0.005, -0.038, 0.363, 0.445, 0.450, 0.350, 0.745, -0.165, 0.294, 0.747, -0.050, 0.474, 1.165, 0.656, 0.503, 0.092, 0.238, 0.042, -0.658, -0.310, 0.296, 0.091, 0.609, -0.349, 0.918, 0.442, -0.640, -0.027, 0.024, 0.718, 0.029, -0.478, 0.694, -0.388, 0.519, 0.265, 0.050, 0.342, 0.136, 0.407, 0.968, -0.263, -0.239, -0.092, 0.169, 0.495, -0.102, 0.232, -0.033, 0.590, 0.074, 1.050, 0.446, 0.371, -0.595, 0.691, 0.679, -0.349, -0.229, 0.481, 0.107, 0.177, 0.635, 0.092, 0.051, 0.260, 0.263, -0.953, 0.415, 0.023, 0.483, 0.201, 0.274, -0.424, 0.206, 0.130, -0.105, 0.216, 0.310, 0.421, 0.284, 0.187, -0.171, -0.053, 0.557, 0.052, -0.052, 0.340, 0.183, 0.909, -0.526, -0.082, 0.483, 0.053, 0.522, 0.096, -0.046, 0.328, -0.232, 0.323, 0.493, -0.457, 0.302, 0.206, 0.429, -0.317, 0.796, 0.404, 0.528, 0.172, 0.106, 0.181};


		for(int i=0; i<startingPoint.length; i++){
			startingPoint[i] = random.nextGaussian();
			System.out.printf("%.3f ", startingPoint[i]);
		};
		System.out.println();
//		printGradientAt(logLikelihood, startingPoint);
//		System.out.println();
//		System.out.println("Starting point:");
//		startingPoint[0] += 0.001;
//		for(int i=0; i<startingPoint.length; i++){
//			startingPoint[i] = 1;
//			System.out.printf("%.3f ",startingPoint[i]);
//		}
//		System.out.println();
//		printGradientAt(logLikelihood, startingPoint);
		
		long startTime = System.currentTimeMillis();
		if (useSGD) {
			GradientDescentMinimizer sgdMinimizer = new GradientDescentMinimizer();
			sgdMinimizer.setType(learnType);
			sgdMinimizer.setIterations(iterations);
			sgdMinimizer.setLearningRate(learningRate);
			sgdMinimizer.setBatchSize(batchSize);
			weights = sgdMinimizer.minimize(logLikelihood, startingPoint);
		} else {
			Result result = null;
			try {
				System.out.println("Start maximizing Log-Likelihood...");
				// Actually minimize the negation of the actual log-likelihood
				// Which is equivalent to maximizing the actual log-likelihood
				result = minimize(logLikelihood, startingPoint);
			} catch (LBFGSBException e) {
				e.printStackTrace();
			}
			weights = result.point;
		}
		long endTime = System.currentTimeMillis();
		//System.out.println("Total function calls: " + cnt);
		System.out.printf("Done minimizing in %.3fs\n", (endTime-startTime)/1000.0);
		
		System.out.println("Weights:");
		for(int i=0; i<weights.length; i++){
			System.out.printf("%.3f ", weights[i]);
		} 
		System.out.println("Done!");
	}

//	private void printGradientAt(LogLikelihood logLikelihood, double[] startingPoint) {
//		System.out.println("Gradient:");
//		FunctionValues values = logLikelihood.getValues(startingPoint);
//		double[] gradient = negate(values.gradient);
//		for(int i=0; i<gradient.length; i++){
//			System.out.printf("%.3f ", gradient[i]);
//		}
//		System.out.println();
//		System.out.printf("Value: %f\n", -values.functionValue);
//	}
	
	/**
	 * Return a double array with negated values
	 * @param values
	 * @return
	 */
	private double[] negate(double[] values){
		double[] result = new double[values.length];
		for(int i=0; i<values.length; i++){
			result[i] = -values[i];
		}
		return result;
	}

	@Override
	public List<Instance> predict(List<Instance> testData) {
		List<Instance> results = new ArrayList<Instance>();
		for(Instance instance: testData){
			int n = instance.words.size()+2;
			int[][] parentIdx = new int[n][tags.size()];
			double[][] lattice = new double[n][tags.size()];
			lattice[0][tags.get(START)] = 1;
			fillValues(instance, lattice, true, weights, CRF::max, parentIdx);
			
			int wordCount = n-1;
			List<TaggedWord> result = new ArrayList<TaggedWord>();
			int curIdx = parentIdx[wordCount][tags.get(END)];
			for(wordCount = n-2; wordCount >= 1; wordCount--){
				result.add(0, new TaggedWord(instance.words.get(wordCount-1).features(), reverseTags[curIdx]));
				curIdx = parentIdx[wordCount][curIdx];
			}
			results.add(new Instance(result));
		}
		return results;
	}
}