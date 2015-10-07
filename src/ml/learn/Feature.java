package ml.learn;

import java.util.Arrays;
import java.util.List;

public class Feature{
	public String[] featureStrings;
	
	public Feature(String[] featureStrings){
		this.featureStrings = featureStrings;
	}
	
	public Feature(List<String> featureStrings){
		this(featureStrings.toArray(new String[featureStrings.size()]));
	}
	
	public boolean equals(Object o){
		if(o instanceof Feature){
			Feature f = (Feature)o;
			if(featureStrings.length != f.featureStrings.length) return false;
			for(int i=0; i<featureStrings.length; i++){
				if(!featureStrings[i].equals(f.featureStrings[i])) return false;
			}
			return true;
		}
		return false;
	}
	
	public int hashCode(){
		return Arrays.hashCode(featureStrings);
	}
	
	public String toString(){
		return Arrays.asList(featureStrings).toString();
	}
}