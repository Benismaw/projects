package fr.ensimag.deca.context;

import fr.ensimag.deca.tools.SymbolTable.Symbol;
import fr.ensimag.deca.tree.Location;

/**
 * Deca Type (internal representation of the compiler)
 *
 * @author gl16
 * @date 01/01/2026
 */

public abstract class Type {


    /**
     * True if this and otherType represent the same type (in the case of
     * classes, this means they represent the same class).
     */
    public abstract boolean sameType(Type otherType);

    private final Symbol name;

    public Type(Symbol name) {
        this.name = name;
    }

    public Symbol getName() {
        return name;
    }

    @Override
    public String toString() {
        return getName().toString();
    }

    public boolean isClass() {
        return false;
    }

    public boolean isInt() {
        return false;
    }

    public boolean isFloat() {
        return false;
    }

    public boolean isBoolean() {
        return false;
    }

    public boolean isVoid() {
        return false;
    }

    public boolean isString() {
        return false;
    }

    public boolean isNull() {
        return false;
    }

    public boolean isClassOrNull() {
        return false;
    }

    public static boolean assignCompatible(EnvironmentType environmentType, Type t1, Type t2) {
        if (t1.isFloat() && t2.isInt()) {
            return true;
        }

        if (t1.sameType(t2)) {
            return true;
        }

        if (t1.isClass()) {
            // null compatible avec toute les classes
            if (t2.isNull()) {
                return true;
            }
            if (t2.isClass()) {
                return environmentType.subtype(t2,t1);
            }
        }

        return false;
    }

    public static boolean castCompatible(EnvironmentType env, Type t1, Type t2) {
        // T1 et T2 ne doivent pas être void
        if (t1.isVoid() || t2.isVoid()) {
            return false;
        }

        // Compatible dans un sens ou dans l'autre
        return assignCompatible(env, t1, t2) || assignCompatible(env, t2, t1);
    }

    /**
     * Returns the same object, as type ClassType, if possible. Throws
     * ContextualError(errorMessage, l) otherwise.
     *
     * Can be seen as a cast, but throws an explicit contextual error when the
     * cast fails.
     */
    public ClassType asClassType(String errorMessage, Location l)
            throws ContextualError {
        throw new ContextualError(errorMessage, l);
    }

}
