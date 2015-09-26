import nn_runtime_jni.nn_runtime;
import nn_runtime_jni.DoubleVector;
import nn_runtime_jni.IdsVector;
import nn_runtime_jni.IdVector;

public class test {
    static {
        System.loadLibrary("nn_runtime_jni");
    }
    static boolean equals(double u, double v){
        return Math.abs(u-v) <= 0.000001;
    }
    public static void main(String argv[]) {
        String modelFile ="../build/model.bin";
        long handle=nn_runtime.load(modelFile);
        assert(handle>0);
        IdVector ids=new IdVector();
        ids.add(0);
        ids.add(2);
        IdsVector idsInputs=new IdsVector();
        idsInputs.add(ids);
        ids=new IdVector();
        ids.add(2);
        idsInputs.add(ids);
        DoubleVector prediction = nn_runtime.predict(handle,idsInputs);
        assert(prediction.size()==2);        
        double t1=Math.tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1);
        double t2=Math.tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2);
        double o1=Math.exp(t1)/(Math.exp(t1)+Math.exp(t2));
        double o2=Math.exp(t2)/(Math.exp(t1)+Math.exp(t2));
        assert(equals(prediction.get(0),o1));
        assert(equals(prediction.get(1),o2));
    }
}


