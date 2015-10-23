package ml.learn.tree;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import lbfgsb.DifferentiableFunction;
import lbfgsb.FunctionValues;
import lbfgsb.IterationFinishedListener;
import lbfgsb.LBFGSBException;
import lbfgsb.Minimizer;
import lbfgsb.Result;
import ml.learn.object.BinaryTree;
import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;
import ml.learn.util.Common;

public class TreeCRF {
	public boolean useLogSpace = false;
	public boolean useAdditionalFeatures = false;
	
	public class LogLikelihood implements DifferentiableFunction {
		public double[] empiricalDistribution;
		public double regularizationParameter;
		
		public LinkedHashMap<BinaryTree, double[][][]> insides;
		public LinkedHashMap<BinaryTree, double[][][]> outsides;
		
		public LogLikelihood(){
			regularizationParameter = 1.0;
			insides = new LinkedHashMap<BinaryTree, double[][][]>();
			outsides = new LinkedHashMap<BinaryTree, double[][][]>();
			empiricalDistribution = calculateEmpiricalDistribution();
//			for(int i=0; i<cnfRules.size(); i++){
//				System.out.println(String.format("Rule %s: %.0f", reverseRules[i], empiricalDistribution[i]));
//			}
		}
		
		@Override
		public FunctionValues getValues(double[] point) {
			double value = 0.0;
			calculateInsideOutside(point);
			value = Common.sum(calculateEmpiricalDistribution(point)).value;
			for(BinaryTree tree: trainingData){
				int n = tree.terminals.length;
				if(useLogSpace){
					value -= insides.get(tree)[tags.get(ROOT)][0][n-1];	
				} else {
					value -= Math.log(insides.get(tree)[tags.get(ROOT)][0][n-1]);
				}
			}
			value -= regularizationTerm(point);
			double[] gradient = computeGradient(point);
//			System.out.println(String.format("Value=%.3f", value));
//			for(int i=0; i<gradient.length; i++){
//				System.out.printf("%.3f ", gradient[i]);
//			}
//			System.out.println();
//			System.out.println();
			return new FunctionValues(-value, Common.negate(gradient));
		}
		
		/**
		 * Compute lambda^2/(2.sigma^2) for calculating likelihood value
		 * @return
		 */
		private double regularizationTerm(double[] point){
			double result = 0;
			for(int i=0; i<point.length; i++){
				result += Math.pow(point[i], 2);
			}
			result /= 2*Math.pow(regularizationParameter, 2);
			return result;
		}
		
		private double[] calculateEmpiricalDistribution(){
			return calculateEmpiricalDistribution(null);
		}
		
		private double[] calculateEmpiricalDistribution(double[] point){
			double[] result = new double[cnfRules.size()];
			Arrays.fill(result, 0.0);
			for(BinaryTree tree: trainingData){
				calculateEmpiricalDistribution(point, result, tree);
			}
			return result;
		}
		
		private void calculateEmpiricalDistribution(double[] point, double[] result, BinaryTree tree){
			CNFRule rule;
			if(tree.left == null){ // Terminal
				rule = new CNFRule(tree.value.tag(), tree.value.word());
			} else {
				Tag leftTag = tree.value.tag();
				Tag firstRightTag = tree.left.value.tag();
				Tag secondRightTag = tree.right.value.tag();
				rule = new CNFRule(leftTag, firstRightTag, secondRightTag);
				if(useAdditionalFeatures){
					String[] terminals = tree.terminals;
					int n = terminals.length;
					if(leftTag.text.matches("NP") && secondRightTag.text.equals("PP")){
						rule.feature = terminals[n-1];
					}
				}
				calculateEmpiricalDistribution(point, result, tree.left);
				calculateEmpiricalDistribution(point, result, tree.right);
			}
			int ruleIdx = cnfRules.getOrDefault(rule, -1);
			if(ruleIdx != -1){
				if(point != null){
					result[ruleIdx] += point[ruleIdx];
				} else {
					result[ruleIdx] += 1;
				}
			}
		}
		
		private double[] computeGradient(double[] point){
			double[] result = new double[point.length];
			double[] modelDistribution = computeModelDistribution(point);
			double[] regularization = computeGradientOfRegularization(point);
			for(int i=0; i<result.length; i++){
				result[i] = empiricalDistribution[i] - modelDistribution[i] - regularization[i];
			}
			return result;
		}
		
		private double[] computeModelDistribution(double[] point){
			double[] result = new double[cnfRules.size()];
			for(BinaryTree tree: trainingData){
				double[] expectation = calculateExpectedCounts(tree, point);
				for(int i=0; i<expectation.length; i++){
					result[i] += expectation[i];
				}
			}
			return result;
		}
		
		/**
		 * Compute lambda/sigma^2 for calculating regularization in gradient
		 * @param point
		 * @return
		 */
		private double[] computeGradientOfRegularization(double[] point){
			double[] result = new double[cnfRules.size()];
			for(int i=0; i<result.length; i++){
				result[i] = point[i]/Math.pow(regularizationParameter, 2);
			}
			return result;
		}
		
		/**
		 * Calculate the expected counts of each rule for the given tree under the given weights for each CNF rule
		 * @param tree The tree
		 * @param weights The weights for each CNF rules
		 * @return
		 */
		private double[] calculateExpectedCounts(BinaryTree tree, double[] weights){
			String[] terminals = tree.terminals;
			int n = terminals.length;
			double[][][] inside = new double[tags.size()][n][n];
			double[][][] outside = new double[tags.size()][n][n];
			calculateInsideOutside(tree, weights, inside, outside);
			double normalizationTerm = inside[tags.get(ROOT)][0][n-1];
			double[] result = new double[weights.length];
			double value = 0.0;
			for(int ruleIdx=0; ruleIdx<result.length; ruleIdx++){
				CNFRule rule = reverseRules[ruleIdx];
				Tag leftTag = rule.leftSide;
				int leftTagIdx = tags.get(leftTag);
				if(rule.terminal == null){
					Tag firstRight = rule.firstRight;
					Tag secondRight =rule.secondRight;
					int firstRightIdx = tags.get(firstRight);
					int secondRightIdx = tags.get(secondRight);
					for(int i=0; i<n; i++){
						for(int j=i+1; j<n; j++){
							if(rule.feature != null && !rule.feature.equals(terminals[j])){
								continue;
							}
							for(int k=i; k<j; k++){
								if(useLogSpace){
									value = outside[leftTagIdx][i][j];
									value += weights[ruleIdx];
									value += inside[firstRightIdx][i][k];
									value += inside[secondRightIdx][k+1][j];
									value -= normalizationTerm;
									value = Math.exp(value);
								} else {
									value = outside[leftTagIdx][i][j];
									value *= Math.exp(weights[ruleIdx]);
									value *= inside[firstRightIdx][i][k];
									value *= inside[secondRightIdx][k+1][j];
									value /= normalizationTerm;
								}
								result[ruleIdx] += value;
							}
						}
					}
				} else {
					for(int i=0; i<n; i++){
						if(tree.terminals[i].equals(rule.terminal)){
							if(useLogSpace){
								result[ruleIdx] += Math.exp(inside[leftTagIdx][i][i]+outside[leftTagIdx][i][i]-normalizationTerm);
							} else {
								result[ruleIdx] += inside[leftTagIdx][i][i]*outside[leftTagIdx][i][i]/normalizationTerm;
							}
						}
					}
				}
			}
			return result;
		}
		
		private void calculateInsideOutside(double[] point){
			for(BinaryTree tree: trainingData){
				String[] terminals = tree.terminals;
				int n = terminals.length;
				double[][][] inside = new double[cnfRules.size()][n][n];
				double[][][] outside = new double[cnfRules.size()][n][n];
				calculateInsideOutside(tree, point, inside, outside);
				insides.put(tree, inside);
				outsides.put(tree, outside);
			}
		}
		
		/**
		 * Calculate the inside and outside potentials with the Inside-Outside algorithm
		 * @param terminals The list of terminals (i.e., words)
		 * @param weights The weights for each CNF rules
		 * @param inside The inside potential array to be filled
		 * @param outside The outside potential array to be filled
		 */
		private void calculateInsideOutside(BinaryTree tree, double[] weights, double[][][] inside, double[][][] outside){
			String[] terminals = tree.terminals;
			int n = terminals.length;
			calculateInside(terminals, weights, inside, null, null);
			calculateOutside(terminals, weights, inside, outside, n);
		}
		
	}
	
	public static final boolean DEBUG = true;
//	public static final DecimalFormat FORMATTER = new DecimalFormat("####0.00;-#");
	public static final DecimalFormat FORMATTER = new DecimalFormat("+##0.000;-#");
	/** The tag considered to be the root tag **/
	public static final Tag ROOT = Tag.get("ROOT");
	
	public List<BinaryTree> trainingData;
	public Map<CNFRule, Integer> cnfRules;
	
	/** Mapping of tags (non-terminals) to indices **/
	public Map<Tag, Integer> tags;
	/** Mapping from tags to rules containing the tag as the left side **/
	public Map<Tag, Set<CNFRule>> tagToRules;
	/** Index of CNF rules based on the left tag, first tag in the right, and second tag in the right **/
	public Map<Tag, Map<Tag, Map<Tag, CNFRule>>> nonTerminalRuleIndex;
	/** Index of CNF rules based on the left tag and the word on the right **/
	public Map<Tag, Map<String, CNFRule>> terminalRuleIndex;
	/** Index of CNF rules based on the first tag on the right side **/
	public Map<Tag, Set<CNFRule>> firstRightTag;
	/** Index of CNF rules based on the second tag on the right side **/
	public Map<Tag, Set<CNFRule>> secondRightTag;

	/** Index of tags **/
	public Tag[] reverseTags;
	/** Index of rules **/
	public CNFRule[] reverseRules;
	
	public Random random;
	
	public double[] weights;
	
	public TreeCRF(){
		cnfRules = new LinkedHashMap<CNFRule, Integer>();
		tagToRules = new LinkedHashMap<Tag, Set<CNFRule>>();
		nonTerminalRuleIndex = new LinkedHashMap<Tag, Map<Tag, Map<Tag, CNFRule>>>();
		terminalRuleIndex = new LinkedHashMap<Tag, Map<String, CNFRule>>();
		firstRightTag = new LinkedHashMap<Tag, Set<CNFRule>>();
		secondRightTag = new LinkedHashMap<Tag, Set<CNFRule>>();
		tags = new LinkedHashMap<Tag, Integer>();
		random = new Random(0);
	}
	
	private Result minimize(DifferentiableFunction function, double[] startingPoint) throws LBFGSBException{
		Minimizer alg = new Minimizer();
		alg.setNoBounds(startingPoint.length);
		alg.setIterationFinishedListener(new IterationFinishedListener(){
			
			public int i=0;
			public double lastValue = 0.0;
			public long startTime = System.currentTimeMillis();

			@Override
			public boolean iterationFinished(double[] point, double functionValue, double[] gradient) {
				i++;
				double change = -functionValue - lastValue;
				String suffix = "";
				if(lastValue != 0){
					change = 100.0*change/-lastValue;
					suffix = "%";
				}
				System.out.printf("Iteration %d: %.9f (%+.6f%s), elapsed time %.3fs\n", i, -functionValue, change, suffix, (System.currentTimeMillis()-startTime)/1000.0);
				lastValue = -functionValue;
//				gradient = CRF.negate(gradient);
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

	/**
	 * Calculate the inside potential values using dynamic programming
	 * @param terminals List of terminals (i.e., words)
	 * @param weights The weights for each CNF rules
	 * @param inside The inside value array to be filled (size: #tags*n*n, where n is the length of terminals)
	 * @param parent To store parent pointer for decoding
	 * @param bestRule To store best rule for decoding
	 */
	private void calculateInside(String[] terminals, double[] weights, double[][][] inside, int[][][] parent, CNFRule[][][] bestRule){
		boolean DEBUG = false;
		int n = terminals.length;
		for(int i=0; i<inside.length; i++){
			for(int j=0; j<n; j++){
				for(int k=0; k<n; k++){
					if(useLogSpace){
						inside[i][j][k] = Double.NEGATIVE_INFINITY;
					} else {
						inside[i][j][k] = 0.0;
					}
				}
			}
		}
		long startTime=0, endTime=0;
		if(DEBUG) System.out.println("Calculating inside...");
		if(DEBUG){
			System.out.printf("%6s","");
			for(Tag tag: tags.keySet()){
				System.out.printf("%9s",tag);
			}
			System.out.println();
		}
		// Base case alpha(A,i,i) = phi(A,i)
		if(DEBUG) startTime = System.currentTimeMillis();
		for(int i=0; i<n; i++){
			String word = terminals[i];
			if(DEBUG) System.out.printf("%2d %2d: ", i, i);
			for(int tagIdx=0; tagIdx<reverseTags.length; tagIdx++){
				try{
					Tag tag = reverseTags[tagIdx];
					CNFRule rule = terminalRuleIndex.get(tag).get(word);
					int ruleIdx = cnfRules.get(rule);
					if(useLogSpace){
						inside[tagIdx][i][i] = weights[ruleIdx];
					} else {
						inside[tagIdx][i][i] = Math.exp(weights[ruleIdx]);
					}
					parent[tagIdx][i][i] = 0;
					bestRule[tagIdx][i][i] = rule;
				} catch (NullPointerException e){
					continue;
				} finally {
					if(DEBUG) System.out.printf("%8s ", FORMATTER.format(inside[tagIdx][i][i]));
				}
			}
			if(DEBUG) System.out.println();
		}
		// Recursive case alpha(A,i,j) = sum_rules sum_k phi(A->BC,i,k,j)*alpha(B,i,k)*alpha(C,k+1,j)
		double weight;
		
		double value;
		double maxValue = 0.0;
		double sumValue = 0.0;
		CNFRule maxRule = null;
		int maxK = -1;
		double[] values = new double[cnfRules.size()];
		double[] valuesRule = new double[n];
		for(int mainDiagIdx=1; mainDiagIdx<n; mainDiagIdx++){
			for(int diagIdx=0; diagIdx<n-mainDiagIdx; diagIdx++){
				int i = diagIdx;
				int j = mainDiagIdx + diagIdx;
				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
				for(int tagIdx=0; tagIdx<reverseTags.length; tagIdx++){
					Tag tag = reverseTags[tagIdx];
					if(useLogSpace){
						maxValue = Double.NEGATIVE_INFINITY;
						Arrays.fill(values, Double.NaN);
					} else {
						maxValue = 0.0;
					}
					sumValue = 0.0;
					for(CNFRule rule: tagToRules.get(tag)){ // Get non-terminal rules
						int ruleIdx = cnfRules.get(rule);
						if(rule.feature != null && !rule.feature.equals(terminals[j])){
							continue;
						}
						Tag firstRight = rule.firstRight;
						Tag secondRight = rule.secondRight;
						int firstRightIdx = tags.get(firstRight);
						int secondRightIdx = tags.get(secondRight);
						if(useLogSpace){
							weight = weights[ruleIdx];
							Arrays.fill(valuesRule, Double.NaN);
							for(int k=i; k<j; k++){
								value = weight+inside[firstRightIdx][i][k]+inside[secondRightIdx][k+1][j];
								valuesRule[k] = value;
								if(parent != null && value > maxValue){
									maxK = k;
									maxRule = rule;
									maxValue = value;
								}
							}
							values[ruleIdx] = Common.sumInLogSpace(valuesRule).value;
						} else {
							weight = Math.exp(weights[ruleIdx]);
							for(int k=i; k<j; k++){
								value = weight*inside[firstRightIdx][i][k]*inside[secondRightIdx][k+1][j];
								if(parent == null){
									sumValue += value;
								} else if(value > maxValue){
									maxK = k;
									maxRule = rule;
									maxValue = value;
								}
							}
						}
					}
					if(parent == null){
						if(useLogSpace){
							inside[tagIdx][i][j] = Common.sumInLogSpace(values).value;
						} else {
							inside[tagIdx][i][j] = sumValue;
						}
					} else {
						inside[tagIdx][i][j] = maxValue;
					}
					if(parent != null) parent[tagIdx][i][j] = maxK;
					if(bestRule != null) bestRule[tagIdx][i][j] = maxRule;
					if(DEBUG) System.out.printf("%8s ", FORMATTER.format(inside[tagIdx][i][j]));
				}
				if(DEBUG) System.out.println();
			}
		}
		if(DEBUG){
			endTime = System.currentTimeMillis();
			System.out.printf("Finished calculating inside in %.3fs\n", (endTime-startTime)/1000.0);
		}
	}

	/**
	 * Calculate the outside potential values using the dynamic programming
	 * @param weights The weights for each CNF rules
	 * @param inside The inside potential values already calculated
	 * @param outside The outside potential value array to be filled
	 * @param n Length of current sample
	 */
	private void calculateOutside(String[] terminals, double[] weights, double[][][] inside, double[][][] outside, int n) {
		boolean DEBUG = false;
		double value;
		double weight;
		long startTime=0;
		long endTime=0;
		for(int i=0; i<outside.length; i++){
			for(int j=0; j<n; j++){
				for(int k=0; k<n; k++){
					if(useLogSpace){
						outside[i][j][k] = Double.NEGATIVE_INFINITY;
					} else {
						outside[i][j][k] = 0.0;
					}
				}
			}
		}
		if(DEBUG){
			System.out.println("Calculating outside...");
			startTime = System.currentTimeMillis();
		}
		// Base case beta(ROOT,1,n)=1, beta(A,1,n)=0
		int tagRootIdx = tags.get(ROOT);
		for(int i=0; i<tags.size(); i++){
			if(useLogSpace){
				outside[i][0][n-1] = Double.NEGATIVE_INFINITY;
			} else {
				outside[i][0][n-1] = 0.0;
			}
		}
		if(useLogSpace){
			outside[tagRootIdx][0][n-1] = 0.0;
		} else {
			outside[tagRootIdx][0][n-1] = 1.0;
		}
		// Recursive case beta(A,i,j) = sum_B->CA sum_k phi(B->CA,k,i-1,j)*alpha(C,k,i-1)*beta(B,k,j)
		//                             +sum_B->AC sum_k phi(B->AC,i,j,k)  *alpha(C,j+1,k)*beta(B,i,k)
		if(DEBUG){
			System.out.printf("%6s","");
			for(Tag tag: tags.keySet()){
				System.out.printf("%9s",tag);
			}
			System.out.println();
		}
		double[] valuesLeft = new double[cnfRules.size()];
		double valueLeft;
		double[] valuesRight = new double[cnfRules.size()];
		double valueRight;
		double[] valuesRule = new double[n];
		for(int len=n-1; len>=0; len--){
			for(int i=0; i<n-len; i++){
				int j = i+len;
				if(i==0 && j==n-1) continue;
				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
				for(int tagIdx=0; tagIdx<reverseTags.length; tagIdx++){
					Tag tag = reverseTags[tagIdx];
					value = 0.0;
					if(useLogSpace){
						Arrays.fill(valuesLeft, Double.NaN);
					}
					// Outside left: sum_B->CA sum_k phi(B->CA,k,i-1,j)*alpha(C,k,i-1)*beta(B,k,j)
					if(secondRightTag.containsKey(tag)){
						for(CNFRule rule: secondRightTag.get(tag)){ // Rules having "tag" as the second right
							int ruleIdx = cnfRules.get(rule);
							if(rule.feature != null && !rule.feature.equals(terminals[j])){
								continue;
							}
							Tag leftSide = rule.leftSide;
							Tag firstRight = rule.firstRight;
							int leftSideIdx = tags.get(leftSide);
							int firstRightIdx = tags.get(firstRight);
							if(useLogSpace){
								weight = weights[ruleIdx];
								Arrays.fill(valuesRule, Double.NaN);
								for(int k=0; k<i; k++){
									value = weight+inside[firstRightIdx][k][i-1]+outside[leftSideIdx][k][j];
									valuesRule[k] = value;
								}
								valuesLeft[ruleIdx] = Common.sumInLogSpace(valuesRule).value;
							} else {
								weight = Math.exp(weights[ruleIdx]);
								for(int k=0; k<i; k++){
									value += weight*inside[firstRightIdx][k][i-1]*outside[leftSideIdx][k][j];
								}
							}
						}
					}
					
					if(useLogSpace){
						Arrays.fill(valuesRight, Double.NaN);
					}
					// Outside right: sum_B->AC sum_k phi(B->AC,i,j,k)*alpha(C,j+1,k)*beta(B,i,k)
					if(firstRightTag.containsKey(tag)){
						for(CNFRule rule: firstRightTag.get(tag)){ // Rules having "tag" as the first right
							int ruleIdx = cnfRules.get(rule);
							Tag leftSide = rule.leftSide;
							Tag secondRight = rule.secondRight;
							int leftSideIdx = tags.get(leftSide);
							int secondRightIdx = tags.get(secondRight);
							if(useLogSpace){
								weight = weights[ruleIdx];
								Arrays.fill(valuesRule, Double.NaN);
								for(int k=j+1; k<n; k++){
									if(rule.feature != null && !rule.feature.equals(terminals[k])){
										continue;
									}
									value = weight+inside[secondRightIdx][j+1][k]+outside[leftSideIdx][i][k];
									valuesRule[k] = value;
								}
								valuesRight[ruleIdx] = Common.sumInLogSpace(valuesRule).value;
							} else {
								weight = Math.exp(weights[ruleIdx]);
								for(int k=j+1; k<n; k++){
									if(rule.feature != null && !rule.feature.equals(terminals[k])){
										continue;
									}
									value += weight*inside[secondRightIdx][j+1][k]*outside[leftSideIdx][i][k];
								}
							}
						}
					}
					if(useLogSpace){
						valueLeft = Common.sumInLogSpace(valuesLeft).value;
						valueRight = Common.sumInLogSpace(valuesRight).value;
						outside[tagIdx][i][j] = Common.sumInLogSpace(new double[]{valueLeft, valueRight}).value;
					} else {
						outside[tagIdx][i][j] = value;
					}
					if(DEBUG) System.out.printf("%8s ", FORMATTER.format(outside[tagIdx][i][j]));
				}
				if(DEBUG) System.out.println();
			}
		}
		if(DEBUG){
			endTime = System.currentTimeMillis();
			System.out.printf("Finished calculating outside in %.3fs\n", (endTime-startTime)/1000.0);
		}
	}

	public void train(List<BinaryTree> trainingData) {
		this.trainingData = trainingData;
		for(BinaryTree tree: trainingData){
			PCFG.normalizeTree(tree);
			tree.fillTerminals();
			buildFeatures(tree);
		}
		reverseTags = new Tag[tags.size()];
		for(Tag tag: tags.keySet()){
			reverseTags[tags.get(tag)] = tag;
		}
		reverseRules = new CNFRule[cnfRules.size()];
		for(CNFRule rule: cnfRules.keySet()){
			reverseRules[cnfRules.get(rule)] = rule;
		}
		System.out.println(cnfRules);
		System.out.println("Grammar size: "+cnfRules.size());
		System.out.println("#Non-terminals: "+tags.size());

		LogLikelihood logLikelihood = new LogLikelihood();
		double[] startingPoint = new double[cnfRules.size()];
		System.out.println("Starting point:");
		for(int i=0; i<startingPoint.length; i++){
			startingPoint[i] = random.nextGaussian();
//			startingPoint[i] = 1;
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
		System.out.println("Weights:");
		for(int i=0; i<weights.length; i++){
			System.out.printf("%s %.3f\n", reverseRules[i], weights[i]);
		}
		System.out.println();
		System.out.println("Done!");
	}
	
	/**
	 * Extract CNF rules from the tree
	 * @param tree
	 */
	public void buildFeatures(BinaryTree tree){
		TaggedWord value = tree.value;
		if(!tags.containsKey((value.tag()))){
			tags.put(value.tag(), tags.size());
			tagToRules.put(value.tag(), new HashSet<CNFRule>());
		}
		CNFRule rule;
		if(tree.left == null){
			Tag tag = value.tag();
			String word = value.word();
			rule = new CNFRule(tag, word);
			if(!terminalRuleIndex.containsKey(tag)){
				terminalRuleIndex.put(tag, new LinkedHashMap<String, CNFRule>());
			}
			terminalRuleIndex.get(tag).put(word, rule);
		} else {
			Tag tag = value.tag();
			Tag leftTag = tree.left.value.tag();
			Tag rightTag = tree.right.value.tag();
			rule = new CNFRule(tag, leftTag, rightTag);
			if(useAdditionalFeatures){
				String[] terminals = tree.terminals;
				int n = terminals.length;
				if(tag.text.matches("NP") && rightTag.text.equals("PP")){
					rule.feature = terminals[n-1];
				}
			}
			if(!nonTerminalRuleIndex.containsKey(tag)){
				nonTerminalRuleIndex.put(tag, new LinkedHashMap<Tag, Map<Tag, CNFRule>>());
			}
			Map<Tag, Map<Tag, CNFRule>> rightSideMap = nonTerminalRuleIndex.get(tag);
			if(!rightSideMap.containsKey(tree.left.value.tag())){
				rightSideMap.put(leftTag, new LinkedHashMap<Tag, CNFRule>());
			}
			rightSideMap.get(leftTag).put(rightTag, rule);
			buildFeatures(tree.left);
			buildFeatures(tree.right);
			tagToRules.get(value.tag()).add(rule);
			if(!firstRightTag.containsKey(leftTag)){
				firstRightTag.put(leftTag, new HashSet<CNFRule>());
			}
			firstRightTag.get(leftTag).add(rule);
			if(!secondRightTag.containsKey(rightTag)){
				secondRightTag.put(rightTag, new HashSet<CNFRule>());
			}
			secondRightTag.get(rightTag).add(rule);
		}
		if(!cnfRules.containsKey(rule)){
			cnfRules.put(rule, cnfRules.size());
		}
	}

	public BinaryTree predict(BinaryTree testData) {
		if(testData.terminals == null){
			testData.fillTerminals();
		}
		return predict(testData.terminals, weights);
	}

	
	/**
	 * Predict the tree structure based on the weights for the grammar that we have
	 * @param terminals
	 * @param weights
	 * @return
	 */
	public BinaryTree predict(String[] terminals, double[] weights){
		int n = terminals.length;
		double[][][] inside = new double[tags.size()][n][n];
		int[][][] parent = new int[tags.size()][n][n];
		CNFRule[][][] bestRule = new CNFRule[tags.size()][n][n];
		calculateInside(terminals, weights, inside, parent, bestRule);

		BinaryTree result = inferBest(terminals, parent, bestRule, tags.get(ROOT), 0, n-1);
		return result;
	}
	
	/**
	 * Given the terminals (list of words) and the parent pointer with the best rule, 
	 * return the best tree (tree with highest potential) spanning from start to end, rooted at the given tag
	 * @param terminals List of terminals (i.e., words)
	 * @param parent The parent pointer
	 * @param bestRule The best rule in relation with the parent
	 * @param tagIdx The index of the root tag of the tree to be returned
	 * @param start The start index of the tree to be returned
	 * @param end The end index of the tree to be returned
	 * @return
	 */
	private BinaryTree inferBest(String[] terminals, int[][][] parent, CNFRule[][][] bestRule, int tagIdx, int start, int end){
		BinaryTree result = new BinaryTree();
		CNFRule rule = bestRule[tagIdx][start][end];
		if(rule == null){
			result.value = new TaggedWord("null", Tag.get("X"));
			return result;
		}
		if(rule.terminal == null){
			Tag firstRight = rule.firstRight;
			Tag secondRight = rule.secondRight;
			int firstRightIdx = tags.get(firstRight);
			int secondRightIdx = tags.get(secondRight);
			int k = parent[tagIdx][start][end];
			result.left = inferBest(terminals, parent, bestRule, firstRightIdx, start, k);
			result.right = inferBest(terminals, parent, bestRule, secondRightIdx, k+1, end);
			result.value = new TaggedWord("", rule.leftSide);
		} else {
			result.value = new TaggedWord(terminals[start], rule.leftSide);
		}
		return result;
	}
}
