package fr.ensimag.ARM;

/**
 * Immediate operand representing an integer.
 * 
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMImmediateInteger extends ARMDVal {
    private int value;

    public ARMImmediateInteger(int value) {
        super();
        this.value = value;
    }

    @Override
    public String toString() {
        return "#" + value;
    }
}
