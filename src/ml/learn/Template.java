package ml.learn;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Template {
	public String template;
	public int[] relativePos;
	public int[] featureIdx;
	
	private boolean isBigram;
	private String featureFormat;
	
	public Template(String template){
		this.template = template;
		compile();
	}
	
	private void compile(){
		Pattern macro = Pattern.compile("%x\\[(-?\\d+),(\\d+)\\]");
		Matcher matcher = macro.matcher(template.substring(1)); // Exclude the U or B
		int numMacros = 0;
		while(matcher.find()){
			numMacros += 1;
		}
		matcher = macro.matcher(template.substring(1));
		relativePos = new int[numMacros];
		featureIdx = new int[numMacros];
		int idx = 0;
		StringBuffer buf = new StringBuffer();
		isBigram = false;
		if(template.startsWith("B")){
			buf.append("%s|");
			isBigram = true;
		}
		buf.append("%s");
		while(matcher.find()){
			relativePos[idx] = Integer.parseInt(matcher.group(1));
			featureIdx[idx] = Integer.parseInt(matcher.group(2));
			idx += 1;
			matcher.appendReplacement(buf, "%s");
		}
		matcher.appendTail(buf);
		featureFormat = buf.toString();
	}
	
	public Feature getFeature(Instance instance, int position, Tag prevTag, Tag curTag){
		List<String> featureArguments = new ArrayList<String>();
		featureArguments.add(featureFormat);
		if(isBigram){
			featureArguments.add(prevTag.text);
		}
		featureArguments.add(curTag.text);
		for(int i=0; i<relativePos.length; i++){
			featureArguments.add(instance.getFeatureAt(position+relativePos[i], featureIdx[i]));
		}
		return new Feature(featureArguments);
	}
}
