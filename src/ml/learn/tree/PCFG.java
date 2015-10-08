package ml.learn.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ml.learn.object.BinaryTree;
import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;

public class PCFG implements EMCompatibleFunction {
	
	public static boolean DEBUG = false;
	
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
		case "SINV":
		case "SQ":
			replacement = Tag.get("S"); break;
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
			tree.fillArray();
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
		double normalizationTerm = inside[tags.get(Tag.get("S"))][0][n-1];
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
		Tag[][] array = tree.array;
		String[] terminals = tree.terminals;
		int n = array.length;
		for(int i=0; i<inside.length; i++){
			for(int j=0; j<n; j++){
				for(int k=0; k<n; k++){
					inside[i][j][k] = 0.0;
					outside[i][j][k] = 0.0;
				}
			}
		}
		long startTime=0, endTime=0;
		// Base case alpha(A,i,i) = phi(A,i)
		if(DEBUG) System.out.println("Calculating inside...");
		if(DEBUG) startTime = System.currentTimeMillis();
		for(int i=0; i<n; i++){
			Tag tag = array[i][i];
			String word = terminals[i];
			int tagIdx = tags.get(array[i][i]);
			int ruleIdx = cnfRules.get(terminalRuleIndex.get(tag).get(word));
			inside[tagIdx][i][i] = weights[ruleIdx];
		}
		// Recursive case alpha(A,i,j) = sum_rules sum_k phi(A->BC,i,k,j)*alpha(B,i,k)*alpha(C,k+1,j)
		if(DEBUG){
			System.out.printf("%6s","");
			for(Tag tag: tags.keySet()){
				System.out.printf("%7s",tag);
			}
			System.out.println();
		}
		for(int mainDiagIdx=1; mainDiagIdx<n; mainDiagIdx++){
			for(int diagIdx=0; diagIdx<n-mainDiagIdx; diagIdx++){
				int i = diagIdx;
				int j = mainDiagIdx + diagIdx;
				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
				for(Tag tag: tags.keySet()){
					int tagIdx = tags.get(tag);
					for(CNFRule rule: tagToRules.get(tag)){ // Get non-terminal rules
						int ruleIdx = cnfRules.get(rule);
						Tag firstRight = rule.firstRight;
						Tag secondRight = rule.secondRight;
						int firstRightIdx = tags.get(firstRight);
						int secondRightIdx = tags.get(secondRight);
						for(int k=i; k<j; k++){
							inside[tagIdx][i][j] += weights[ruleIdx]*inside[firstRightIdx][i][k]*inside[secondRightIdx][k+1][j];
						}
					}
					if(DEBUG) System.out.printf("%+.3f ", inside[tagIdx][i][j]);
				}
				if(DEBUG) System.out.println();
			}
		}
		if(DEBUG){
			endTime = System.currentTimeMillis();
			System.out.printf("Finished calculating inside in %.3fs\n", (endTime-startTime)/1000.0);
		}

		if(DEBUG){
			System.out.println("Calculating outside...");
			startTime = System.currentTimeMillis();
		}
		// Base case beta(S,1,n)=1, beta(A,1,n)=0
		int tagSIdx = tags.get(Tag.get("S"));
		for(int i=0; i<tags.size(); i++){
			outside[i][0][n-1] = 0.0;
		}
		outside[tagSIdx][0][n-1] = 1.0;
		// Recursive case beta(A,i,j) = sum_B->CA sum_k phi(B->CA,k,i-1,j)*alpha(C,k,i-1)*beta(B,k,j)
		//                             +sum_B->AC sum_k phi(B->AC,i,j,k)  *alpha(C,j+1,k)*beta(B,i,k)
		if(DEBUG){
			System.out.printf("%6s","");
			for(Tag tag: tags.keySet()){
				System.out.printf("%7s",tag);
			}
			System.out.println();
		}
		for(int len=n-1; len>=0; len--){
			for(int i=0; i<n-len; i++){
				int j = i+len;
				if(i==0 && j==n-1) continue;
				if(DEBUG) System.out.printf("%2d %2d: ", i, j);
				for(Tag tag: tags.keySet()){
					int tagIdx = tags.get(tag);
					// Outside left
					for(CNFRule rule: secondRightTag.getOrDefault(tag, new ArrayList<CNFRule>())){ // Rules having "tag" as the second right
						int ruleIdx = cnfRules.get(rule);
						Tag leftSide = rule.leftSide;
						Tag firstRight = rule.firstRight;
						int leftSideIdx = tags.get(leftSide);
						int firstRightIdx = tags.get(firstRight);
						for(int k=0; k<i; k++){
							outside[tagIdx][i][j] += weights[ruleIdx]*inside[firstRightIdx][k][i-1]*outside[leftSideIdx][k][j];
						}
					}
					// Outside right
					for(CNFRule rule: firstRightTag.getOrDefault(tag, new ArrayList<CNFRule>())){ // Rules having "tag" as the first right
						int ruleIdx = cnfRules.get(rule);
						Tag leftSide = rule.leftSide;
						Tag secondRight = rule.secondRight;
						int leftSideIdx = tags.get(leftSide);
						int secondRightIdx = tags.get(secondRight);
						for(int k=j+1; k<n; k++){
							outside[tagIdx][i][j] += weights[ruleIdx]*inside[secondRightIdx][j+1][k]*outside[leftSideIdx][i][k];
						}
					}
					if(DEBUG) System.out.printf("%+.3f ", outside[tagIdx][i][j]);
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

}
