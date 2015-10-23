package ml.learn.tree;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.learn.object.BinaryTree;
import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;

/**
 * A class compatible for EM Algorithm for the application of PCFG.
 * In addition to implementing the functions defined by {@link EMCompatibleFunction}, this defines 
 * another method {@link #predict(String[], double[])} to decode the tree out of the given terminals
 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
 *
 */
public class PCFG implements EMCompatibleFunction {
	
	public static final boolean DEBUG = false;
	public static final DecimalFormat FORMATTER = new DecimalFormat("####0.00;-#");
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
	
	/** The probability for each rule under this PCFG model */
	public double[] weights;
	
	/**
	 * Initialize a PCFG by extracting CNF rules from the list of binary trees
	 * @param trainingData
	 */
	public PCFG(List<BinaryTree> trainingData){
		this.trainingData = trainingData;
		cnfRules = new LinkedHashMap<CNFRule, Integer>();
		tagToRules = new LinkedHashMap<Tag, Set<CNFRule>>();
		nonTerminalRuleIndex = new LinkedHashMap<Tag, Map<Tag, Map<Tag, CNFRule>>>();
		terminalRuleIndex = new LinkedHashMap<Tag, Map<String, CNFRule>>();
		firstRightTag = new LinkedHashMap<Tag, Set<CNFRule>>();
		secondRightTag = new LinkedHashMap<Tag, Set<CNFRule>>();
		tags = new LinkedHashMap<Tag, Integer>();
		for(BinaryTree tree: trainingData){
			normalizeTree(tree);
			insertRules(tree);
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
	}
	
	/**
	 * Normalize tree by coalescing some tags
	 * @param tree
	 */
	public static void normalizeTree(BinaryTree tree){
		Tag tag = tree.value.tag();
		Tag replacement;
		switch(tag.text){
//		case "QP":
//			replacement = Tag.get("CD"); break;
//		case "RBS":
//		case "RBR":
//			replacement = Tag.get("RB"); break;
//		case "SINV":
//		case "SQ":
//			replacement = Tag.get("S"); break;
//		case "SBARQ":
//			replacement = Tag.get("SBAR"); break;
		case "NX":
		case "NAC":
			replacement = Tag.get("NP"); break;
		default:
			replacement = tag;
		}
		tree.value.setTag(replacement);
		if(tree.left != null){
			normalizeTree(tree.left);
			normalizeTree(tree.right);
		}
	}
	
	/**
	 * Extract CNF rules from the tree
	 * @param tree
	 */
	private void insertRules(BinaryTree tree){
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
			if(!nonTerminalRuleIndex.containsKey(tag)){
				nonTerminalRuleIndex.put(tag, new LinkedHashMap<Tag, Map<Tag, CNFRule>>());
			}
			Map<Tag, Map<Tag, CNFRule>> rightSideMap = nonTerminalRuleIndex.get(tag);
			if(!rightSideMap.containsKey(tree.left.value.tag())){
				rightSideMap.put(leftTag, new LinkedHashMap<Tag, CNFRule>());
			}
			rightSideMap.get(leftTag).put(rightTag, rule);
			insertRules(tree.left);
			insertRules(tree.right);
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
	
	@Override
	public int numParams(){
		return cnfRules.size();
	}

	@Override
	public double[] expectation(double[] weights) {
		double[] result = new double[weights.length];
		Arrays.fill(result, 0.0);
		for(BinaryTree tree: trainingData){
			tree.fillTerminals();
			double[] count = calculateExpectedCounts(tree, weights);
			for(int ruleIdx=0; ruleIdx<cnfRules.size(); ruleIdx++){
				result[ruleIdx] += count[ruleIdx];
			}
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
		int n = tree.terminals.length;
		double[][][] inside = new double[tags.size()][n][n];
		double[][][] outside = new double[tags.size()][n][n];
		calculateInsideOutside(tree.terminals, weights, inside, outside);
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
						for(int k=i; k<j; k++){
							value = outside[leftTagIdx][i][j];
							value *= weights[ruleIdx];
							value *= inside[firstRightIdx][i][k];
							value *= inside[secondRightIdx][k+1][j];
							result[ruleIdx] += value;
						}
					}
				}
				result[ruleIdx] /= normalizationTerm;
			} else {
				for(int i=0; i<n; i++){
					if(tree.terminals[i].equals(rule.terminal)){
						result[ruleIdx] += inside[leftTagIdx][i][i]*outside[leftTagIdx][i][i];
					}
				}
				result[ruleIdx] /= normalizationTerm;
			}
		}
		return result;
	}
	
	/**
	 * Calculate the inside and outside potentials with the Inside-Outside algorithm
	 * @param terminals The list of terminals (i.e., words)
	 * @param weights The weights for each CNF rules
	 * @param inside The inside potential array to be filled
	 * @param outside The outisde potential array to be filled
	 */
	private void calculateInsideOutside(String[] terminals, double[] weights, double[][][] inside, double[][][] outside){
		int n = terminals.length;
		for(int i=0; i<inside.length; i++){
			for(int j=0; j<n; j++){
				for(int k=0; k<n; k++){
					inside[i][j][k] = 0.0;
					outside[i][j][k] = 0.0;
				}
			}
		}
		calculateInside(terminals, weights, inside, null, null);
		calculateOutside(weights, inside, outside, n);
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
					inside[i][j][k] = 0.0;
				}
			}
		}
		long startTime=0, endTime=0;
		// Base case alpha(A,i,i) = phi(A,i)
		if(DEBUG) System.out.println("Calculating inside...");
		if(DEBUG) startTime = System.currentTimeMillis();
		for(int i=0; i<n; i++){
			String word = terminals[i];
			for(int tagIdx=0; tagIdx<reverseTags.length; tagIdx++){
				try{
					Tag tag = reverseTags[tagIdx];
					CNFRule rule = terminalRuleIndex.get(tag).get(word);
					int ruleIdx = cnfRules.get(rule);
					inside[tagIdx][i][i] = weights[ruleIdx];
					parent[tagIdx][i][i] = 0;
					bestRule[tagIdx][i][i] = rule;
				} catch (NullPointerException e){
					continue;
				}
			}
		}
		// Recursive case alpha(A,i,j) = sum_rules sum_k phi(A->BC,i,k,j)*alpha(B,i,k)*alpha(C,k+1,j)
//		if(DEBUG){
//			System.out.printf("%6s","");
//			for(Tag tag: tags.keySet()){
//				System.out.printf("%9s",tag);
//			}
//			System.out.println();
//		}
		double weight;
		
		double value;
		double maxValue = 0.0;
		double sumValue = 0.0;
		CNFRule maxRule = null;
		int maxK = -1;
		for(int mainDiagIdx=1; mainDiagIdx<n; mainDiagIdx++){
			for(int diagIdx=0; diagIdx<n-mainDiagIdx; diagIdx++){
				int i = diagIdx;
				int j = mainDiagIdx + diagIdx;
//				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
				for(int tagIdx=0; tagIdx<reverseTags.length; tagIdx++){
					Tag tag = reverseTags[tagIdx];
					maxValue = 0.0;
					sumValue = 0.0;
					for(CNFRule rule: tagToRules.get(tag)){ // Get non-terminal rules
						int ruleIdx = cnfRules.get(rule);
						Tag firstRight = rule.firstRight;
						Tag secondRight = rule.secondRight;
						int firstRightIdx = tags.get(firstRight);
						int secondRightIdx = tags.get(secondRight);
						weight = weights[ruleIdx];
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
					if(parent == null){
						inside[tagIdx][i][j] = sumValue;
					} else {
						inside[tagIdx][i][j] = maxValue;
					}
					if(parent != null) parent[tagIdx][i][j] = maxK;
					if(bestRule != null) bestRule[tagIdx][i][j] = maxRule;
//					if(DEBUG) System.out.printf("%8s ", FORMATTER.format(inside[tagIdx][i][j]));
				}
//				if(DEBUG) System.out.println();
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
	private void calculateOutside(double[] weights, double[][][] inside, double[][][] outside, int n) {
		double value;
		double weight;
		long startTime;
		long endTime;
		if(DEBUG){
			System.out.println("Calculating outside...");
			startTime = System.currentTimeMillis();
		}
		// Base case beta(ROOT,1,n)=1, beta(A,1,n)=0
		int tagRootIdx = tags.get(ROOT);
		for(int i=0; i<tags.size(); i++){
			outside[i][0][n-1] = 0.0;
		}
		outside[tagRootIdx][0][n-1] = 1.0;
		// Recursive case beta(A,i,j) = sum_B->CA sum_k phi(B->CA,k,i-1,j)*alpha(C,k,i-1)*beta(B,k,j)
		//                             +sum_B->AC sum_k phi(B->AC,i,j,k)  *alpha(C,j+1,k)*beta(B,i,k)
//		if(DEBUG){
//			System.out.printf("%6s","");
//			for(Tag tag: tags.keySet()){
//				System.out.printf("%9s",tag);
//			}
//			System.out.println();
//		}
		for(int len=n-1; len>=0; len--){
			for(int i=0; i<n-len; i++){
				int j = i+len;
				if(i==0 && j==n-1) continue;
//				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
				for(int tagIdx=0; tagIdx<reverseTags.length; tagIdx++){
					Tag tag = reverseTags[tagIdx];
					value = 0.0;
					// Outside left: sum_B->CA sum_k phi(B->CA,k,i-1,j)*alpha(C,k,i-1)*beta(B,k,j)
					if(secondRightTag.containsKey(tag)){
						for(CNFRule rule: secondRightTag.get(tag)){ // Rules having "tag" as the second right
							int ruleIdx = cnfRules.get(rule);
							Tag leftSide = rule.leftSide;
							Tag firstRight = rule.firstRight;
							int leftSideIdx = tags.get(leftSide);
							int firstRightIdx = tags.get(firstRight);
							weight = weights[ruleIdx];
							for(int k=0; k<i; k++){
								value += weight*inside[firstRightIdx][k][i-1]*outside[leftSideIdx][k][j];
							}
						}
					}
					// Outside right: sum_B->AC sum_k phi(B->AC,i,j,k)*alpha(C,j+1,k)*beta(B,i,k)
					if(firstRightTag.containsKey(tag)){
						for(CNFRule rule: firstRightTag.get(tag)){ // Rules having "tag" as the first right
							int ruleIdx = cnfRules.get(rule);
							Tag leftSide = rule.leftSide;
							Tag secondRight = rule.secondRight;
							int leftSideIdx = tags.get(leftSide);
							int secondRightIdx = tags.get(secondRight);
							weight = weights[ruleIdx];
							for(int k=j+1; k<n; k++){
								value += weight*inside[secondRightIdx][j+1][k]*outside[leftSideIdx][i][k];
							}
						}
					}
					outside[tagIdx][i][j] = value;
//					if(DEBUG) System.out.printf("%8s ", FORMATTER.format(outside[tagIdx][i][j]));
				}
//				if(DEBUG) System.out.println();
			}
		}
		if(DEBUG){
			endTime = System.currentTimeMillis();
			System.out.printf("Finished calculating outside in %.3fs\n", (endTime-startTime)/1000.0);
		}
	}

	@Override
	public double[] maximize(double[] expectations) {
		double sum;
		double[] result = new double[expectations.length];
		for(Tag tag: tags.keySet()){
			Collection<CNFRule> rulesWithStartingTag = new ArrayList<CNFRule>();
			rulesWithStartingTag.addAll(tagToRules.get(tag));
			rulesWithStartingTag.addAll(terminalRuleIndex.getOrDefault(tag, new LinkedHashMap<String, CNFRule>()).values());
			sum = 0.0;
			for(CNFRule rule: rulesWithStartingTag){
				sum += expectations[cnfRules.get(rule)];
			}
			for(CNFRule rule: rulesWithStartingTag){
				int idx = cnfRules.get(rule);
				result[idx] = expectations[idx] / sum;
			}
		}
		return result;
	}
	
	/**
	 * Predict the tree structure based on the weights for the grammar that we have
	 * @param terminals
	 * @param weights
	 * @return
	 */
	public BinaryTree predict(BinaryTree tree){
		String[] terminals = tree.terminals;
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
	
	@Override
	public void setParams(double[] params){
		weights = params;
	}
}
