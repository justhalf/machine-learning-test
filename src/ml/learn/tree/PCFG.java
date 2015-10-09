package ml.learn.tree;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ml.learn.object.BinaryTree;
import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;

public class PCFG implements EMCompatibleFunction {
	
	public static final boolean DEBUG = false;
	public static final DecimalFormat FORMATTER = new DecimalFormat("+0.0E0;-#");
	public static final Tag ROOT = Tag.get("ROOT");
	
	public List<BinaryTree> trainingData;
	public Map<CNFRule, Integer> cnfRules;
	
	public Map<Tag, Integer> tags;
	public Map<Tag, List<CNFRule>> tagToRules;
	public Map<Tag, Map<Tag, Map<Tag, CNFRule>>> nonTerminalRuleIndex;
	public Map<Tag, Map<String, CNFRule>> terminalRuleIndex;
	public Map<Tag, List<CNFRule>> firstRightTag;
	public Map<Tag, List<CNFRule>> secondRightTag;

	public Tag[] reverseTags;
	public CNFRule[] reverseRules;
	
	public PCFG(List<BinaryTree> trainingData){
		this.trainingData = trainingData;
		cnfRules = new LinkedHashMap<CNFRule, Integer>();
		tagToRules = new LinkedHashMap<Tag, List<CNFRule>>();
		nonTerminalRuleIndex = new LinkedHashMap<Tag, Map<Tag, Map<Tag, CNFRule>>>();
		terminalRuleIndex = new LinkedHashMap<Tag, Map<String, CNFRule>>();
		firstRightTag = new LinkedHashMap<Tag, List<CNFRule>>();
		secondRightTag = new LinkedHashMap<Tag, List<CNFRule>>();
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
	
	private void normalizeTree(BinaryTree tree){
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
	
	private void insertRules(BinaryTree tree){
		TaggedWord value = tree.value;
		if(!tags.containsKey((value.tag()))){
			tags.put(value.tag(), tags.size());
			tagToRules.put(value.tag(), new ArrayList<CNFRule>());
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
				firstRightTag.put(leftTag, new ArrayList<CNFRule>());
			}
			firstRightTag.get(leftTag).add(rule);
			if(!secondRightTag.containsKey(rightTag)){
				secondRightTag.put(rightTag, new ArrayList<CNFRule>());
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
	
	private double[] calculateExpectedCounts(BinaryTree tree, double[] weights){
		int n = tree.terminals.length;
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
	
	private void calculateInsideOutside(BinaryTree tree, double[] weights, double[][][] inside, double[][][] outside){
		String[] terminals = tree.terminals;
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
		if(DEBUG){
			System.out.printf("%6s","");
			for(Tag tag: tags.keySet()){
				System.out.printf("%9s",tag);
			}
			System.out.println();
		}
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
				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
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
		if(DEBUG){
			System.out.printf("%6s","");
			for(Tag tag: tags.keySet()){
				System.out.printf("%9s",tag);
			}
			System.out.println();
		}
		for(int len=n-1; len>=0; len--){
			for(int i=0; i<n-len; i++){
				int j = i+len;
				if(i==0 && j==n-1) continue;
				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
				for(int tagIdx=0; tagIdx<reverseTags.length; tagIdx++){
					Tag tag = reverseTags[tagIdx];
					value = 0.0;
					// Outside left
					for(CNFRule rule: secondRightTag.getOrDefault(tag, new ArrayList<CNFRule>())){ // Rules having "tag" as the second right
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
					// Outside right
					for(CNFRule rule: firstRightTag.getOrDefault(tag, new ArrayList<CNFRule>())){ // Rules having "tag" as the first right
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
					outside[tagIdx][i][j] = value;
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

	@Override
	public double[] maximize(double[] expectations) {
		double sum = 0.0;
		for(int i=0; i<expectations.length; i++){
			sum += expectations[i];
		}
		double[] result = new double[expectations.length];
		for(int i=0; i<expectations.length; i++){
			result[i] = expectations[i] / sum;
		}
		return result;
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
