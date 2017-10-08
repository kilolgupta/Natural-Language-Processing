import java.io.*;
import java.util.*;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.io.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;

public class coreferenceResolutionCaseless {
    public static void main(String[] args) throws Exception {

        PrintWriter out = new PrintWriter("outFile.txt");
        PrintWriter xmlOut = new PrintWriter("outFile.xml");

        Annotation document = new Annotation("barack obama went to russia. he was impressed by it.");


        Properties caselessProps = new Properties();
        caselessProps.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        caselessProps.put( "pos.model", "english-caseless-left3words-distsim.tagger");
        caselessProps.put( "parse.model", "englishPCFG.caseless.ser.gz" );
        caselessProps.put( "ner.model.3class", "english.all.3class.caseless.distsim.crf.ser.gz");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(caselessProps);
        Annotation annotation = new Annotation(document);
        pipeline.annotate(annotation);

        pipeline.prettyPrint(annotation, out);
        if (xmlOut != null) {
            pipeline.xmlPrint(annotation, xmlOut);
        }

        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        if (sentences != null && !sentences.isEmpty()) {
            System.out.println("Coreference information");
            Map<Integer, CorefChain> corefChains =
                    annotation.get(CorefCoreAnnotations.CorefChainAnnotation.class);
            if (corefChains == null) {
                return;
            }
            for (Map.Entry<Integer, CorefChain> entry : corefChains.entrySet()) {
                System.out.println("Chain " + entry.getKey());
                for (CorefChain.CorefMention m : entry.getValue().getMentionsInTextualOrder()) {
                    // We need to subtract one since the indices count from 1 but the Lists start from 0
                    List<CoreLabel> tokens = sentences.get(m.sentNum - 1).get(CoreAnnotations.TokensAnnotation.class);
                    // We subtract two for end: one for 0-based indexing, and one because we want last token of mention not one following.
                    System.out.println("  " + m + ", i.e., 0-based character offsets [" + tokens.get(m.startIndex - 1).beginPosition() +
                            ", " + tokens.get(m.endIndex - 2).endPosition() + ")");
                }
            }
            System.out.println();
        }
        IOUtils.closeIgnoringExceptions(out);
        IOUtils.closeIgnoringExceptions(xmlOut);
    }
}