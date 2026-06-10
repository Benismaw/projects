package fr.ensimag.ARM;

public class ARMImmediate extends ARMOperand {
    private int value;

    public ARMImmediate(int value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "#" + value;
    }
    
}
