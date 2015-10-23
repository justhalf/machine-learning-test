package ml.learn.tree;

import java.util.Arrays;

import ml.learn.object.Tag;

public class CNFRule {
	public Tag leftSide;
	public Tag firstRight;
	public Tag secondRight;
	
	public String terminal;
	
	public String feature;
	
	public CNFRule(Tag leftSide, Tag firstRight, Tag secondRight){
		this.leftSide = leftSide;
		this.firstRight = firstRight;
		this.secondRight = secondRight;
	}
	
	public CNFRule(Tag leftSide, String terminal){
		this.leftSide = leftSide;
		this.terminal = terminal;
	}
	
	public String toString(){
		if(terminal == null){
			if(feature != null){
				return leftSide+"->"+firstRight+" "+secondRight+"("+feature+")";
			} else {
				return leftSide+"->"+firstRight+" "+secondRight;
			}
		} else {
			return leftSide+"->"+terminal;
		}
	}
	
	public boolean equals(Object o){
		if(o instanceof CNFRule){
			CNFRule rule = (CNFRule)o;
			if(leftSide != rule.leftSide) return false;
			if(firstRight != rule.firstRight) return false;
			if(secondRight != rule.secondRight) return false;
			if(terminal == null){
				if(rule.terminal != null) return false;
			} else {
				if(!terminal.equals(rule.terminal)) return false;
			}
			if(feature == null){
				return rule.feature == null;
			} else {
				return feature.equals(rule.feature);
			}
		}
		return false;
	}
	
	public int hashCode(){
		return Arrays.hashCode(new Object[]{leftSide, firstRight, secondRight, terminal, feature});
	}
}
