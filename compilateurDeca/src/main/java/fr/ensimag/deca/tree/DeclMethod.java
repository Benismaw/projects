//classe pour les methodes 
package fr.ensimag.deca.tree;
import java.io.PrintStream;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.deca.context.*;
import fr.ensimag.deca.context.MethodDefinition;
import fr.ensimag.deca.tools.SymbolTable;
import fr.ensimag.ima.pseudocode.*;
import fr.ensimag.ima.pseudocode.instructions.*;

public class DeclMethod extends AbstractDeclMethod{

    
    private AbstractIdentifier typeRetour;
    private AbstractIdentifier nomMethod;
    private ListDeclParam listeParam;
    private AbstractMethodBody methodBody;

    public DeclMethod(AbstractIdentifier typeRetour,AbstractIdentifier nomMethod,ListDeclParam listeParam,AbstractMethodBody methodBody){
        this.typeRetour=typeRetour;
        this.nomMethod=nomMethod;
        this.listeParam=listeParam;
        this.methodBody=methodBody;
    }
    //getters
    public AbstractIdentifier getTypeRetour(){
        return this.typeRetour;
    }

    @Override
    public AbstractIdentifier getNomMethod(){
        return this.nomMethod;

    }
    @Override
    public MethodDefinition getMethodDefinition() {
        return (MethodDefinition) this.nomMethod.getDefinition();
    }

    public ListDeclParam getListeParam(){
        return this.listeParam;
        
    }

    public AbstractMethodBody getMethodBody(){
        return this.methodBody;
        
    }
     protected  void verifyMethodMembers(DecacCompiler compiler,ClassDefinition superClass,
            ClassDefinition currentClass) throws ContextualError{
                 //verifier le type (pas void) et déclarer le champs 
                 Type typeRet=this.typeRetour.verifyType(compiler);
                //construction de la signature
                Signature sig=this.listeParam.verifyListParam(compiler);

                //gestion de la redefinition (override)
                ExpDefinition superDef=superClass.getMembers().get(this.nomMethod.getName());
                int index;
                if (superDef != null && superDef.isMethod()) {
                     MethodDefinition superMethod = superDef.asMethodDefinition("Erreur redéfinition", getLocation());
            
                    // Vérification stricte
                    if (sig.size() != superMethod.getSignature().size()) {
                        throw new ContextualError("Signature différente de la super-méthode", getLocation());
                    }
                    for (int i = 0; i < sig.size(); i++) {
                        if (!sig.paramNumber(i).sameType(superMethod.getSignature().paramNumber(i))) {
                            throw new ContextualError("Signature différente de la super-méthode", getLocation());
                        }
                    }

                    // Type de retour doit être un sous-type 
                    if (!compiler.environmentType.subtype(typeRet, superMethod.getType())) {
                        throw new ContextualError("Type de retour incompatible avec l'héritage", getLocation());
                    }
                    // Redéfinition : on garde le même index 
                    index = superMethod.getIndex();


                } else if (superDef != null && superDef.isField()) {
                    throw new ContextualError("Une méthode ne peut pas redéfinir un champ", getLocation()); 
                } else {
                    // Nouvelle méthode : on incrémente le compteur de la classe
                    index = currentClass.incNumberOfMethods();
                }

                // Création de la définition et décoration
                MethodDefinition methodDef = new MethodDefinition(typeRet, getLocation(), sig, index);
                String nomLabel = "code." + currentClass.getType().getName().getName() + "." + this.nomMethod.getName().getName();
                methodDef.setLabel(new Label(nomLabel));
                try {
                    currentClass.getMembers().declare(this.nomMethod.getName(), methodDef);
                } catch (EnvironmentExp.DoubleDefException e) {
                    throw new ContextualError("Méthode déjà définie dans cette classe", getLocation());
                }
                this.nomMethod.setDefinition(methodDef);
            }
            
    //passe 3 
    protected  void verifyMethodBody(DecacCompiler compiler,
            ClassDefinition currentClass) throws ContextualError{
        // Creation de l'environnement local
        EnvironmentExp localEnv = new EnvironmentExp(currentClass.getMembers());

        // Ajout de this dans l'environnement local
        SymbolTable.Symbol thisSymbol = compiler.symbolTable.create("this");
        ParamDefinition thisDef = new ParamDefinition(
                currentClass.getType(),
                this.getLocation()
        );
        // On lui donne son adresse physique immédiate
        thisDef.setOperand(new fr.ensimag.ima.pseudocode.RegisterOffset(-2, Register.LB));

        try {
            localEnv.declare(thisSymbol, thisDef);
        } catch (EnvironmentExp.DoubleDefException e) {
            // Ne devrai pas arriver
        }

        // Ajout des parametres dans l'environnement
        this.listeParam.verifyListParamMembers(compiler, localEnv);

        Type returnType=this.typeRetour.verifyType(compiler);

        // On verifie le corps de la methode
        this.methodBody.verifyMethodBody(compiler, localEnv, currentClass, returnType);
    }

    @Override
    public void codeGenDeclMethod(DecacCompiler compiler, String className) {
        String methodName = this.nomMethod.getName().getName();
        Label startLabel = this.nomMethod.getMethodDefinition().getLabel();
        Label endLabel = new Label("fin." + className + "." + methodName);
        compiler.setCurrentMethodEndLabel(endLabel);

        // On place le label de début
        compiler.addLabel(startLabel);

        // On mémorise l'index actuel dans le programme pour ajouter ensuite le test de pile
        int preambleIndex = compiler.getProgram().size();

        // Reset des compteurs pour analyser ce corps spécifiquement
        compiler.getRegManager().reset();
        compiler.getStackManager().reset();

        // Génération du corps
        methodBody.codeGenMethodBody(compiler);

        // Calcul des ressources utilisées
        int nbLocals = 0;
        if (methodBody instanceof MethodBody) {
            nbLocals = ((MethodBody) methodBody).getNumLocals();
        }

        // Registres max utilisés
        int maxRegUsed = compiler.getRegManager().getMaxRegUsed();

        // Registres à sauvegarder
        int nbSavedRegs = Math.max(0, maxRegUsed - 1);

        // Pile temporaire max atteinte par les expressions (Push/Pop temporaires)
        int maxTempStack = compiler.getStackManager().getMaxStack();

        // Formule complète de d pour TSTO
        int d = nbSavedRegs + nbLocals + maxTempStack;

        // On incrémente 'offset' pour insérer les instructions dans le bon ordre.
        int offset = 0;

        // Test de la pile
        compiler.getProgram().addInstructionAt(preambleIndex + offset++, new TSTO(d));
        compiler.getProgram().addInstructionAt(preambleIndex + offset++, new BOV(new Label("pile_pleine")));

        // B. Sauvegarde des registres (PUSH R2 ... PUSH Rmax)
        for (int i = 2; i <= maxRegUsed; i++) {
            compiler.getProgram().addInstructionAt(preambleIndex + offset++, new PUSH(Register.getR(i)));
        }

        // Réservation espace pour variables locales
        if (nbLocals > 0) {
            compiler.getProgram().addInstructionAt(preambleIndex + offset++, new ADDSP(nbLocals));
        }

        // Fin de méthode
        // Vérification de sortie sans return (pour les méthodes non void)
        if (!this.nomMethod.getMethodDefinition().getType().isVoid()) {
            compiler.addInstruction(new WSTR("Erreur : sortie de methode " + methodName + " sans return"));
            compiler.addInstruction(new WNL());
            compiler.addInstruction(new ERROR());
        }

        compiler.addLabel(endLabel);

        // Nettoyage de la pile et Restauration

        // Libération des locales
        if (nbLocals > 0) {
            compiler.addInstruction(new SUBSP(nbLocals));
        }

        // Restauration des registres (Ordre inverse du PUSH)
        for (int i = maxRegUsed; i >= 2; i--) {
            compiler.addInstruction(new POP(Register.getR(i)));
        }

        // Retour
        compiler.addInstruction(new RTS());
    }

        @Override
    public void decompile(IndentPrintStream s){
        typeRetour.decompile(s);
        s.print(" ");
        nomMethod.decompile(s);
        s.print("(");
        listeParam.decompile(s);
          s.print(") ");
        methodBody.decompile(s);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
            typeRetour.prettyPrint(s, prefix, false);
            nomMethod.prettyPrint(s, prefix, false);
            listeParam.prettyPrint(s, prefix, false);
            methodBody.prettyPrint(s, prefix, true);
    }

    @Override
    protected void iterChildren(TreeFunction f) {
            typeRetour.iter(f);
            nomMethod.iter(f);
            listeParam.iter(f);
            methodBody.iter(f);
    }
}
