package fr.ensimag.ARM;

import org.apache.commons.lang.Validate;

/**
 * Label used as operand
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMLabelOperand extends ARMDVal {

    private ARMLabel label;
    
     public ARMLabel getLabel() {
        return label;
    }

    public ARMLabelOperand(ARMLabel label) {
        super();
        Validate.notNull(label);
        this.label = label;
    }

    @Override
    public String toString() {
        return label.getName();
    }

}   
