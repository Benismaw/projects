package fr.ensimag.ARM;

/**
 * Immediate operand containing a float value.
 * 
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMImmediateFloat extends ARMDVal {
    private float value;

    public ARMImmediateFloat(float value) {
        super();
        this.value = value;
    }

    @Override
    public String toString() {
        return "#" + Float.toHexString(value);
    }
}
