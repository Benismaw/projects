package fr.ensimag.ARM;

/**
 * Immediate operand representing a string.
 * 
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMImmediateString extends ARMOperand {
    private String value;

    public ARMImmediateString(String value) {
        super();
        this.value = value;
    }

    @Override
    public String toString() {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }
}
