// Fill out your copyright notice in the Description page of Project Settings.


#include "PoleController.h"

// Sets default values for this component's properties
UPoleController::UPoleController()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UPoleController::BeginPlay()
{
	Super::BeginPlay();

	InputComponent = GetOwner()->InputComponent;
	if (InputComponent) {
		InputComponent->BindAxis(TEXT("X"), this, &UPoleController::Move_XAxis);
	}
	
}

void UPoleController::Move_XAxis(float AxisValue)
{
	CurrMotorSpeed += AxisValue * MotorPower;
	CurrMotorSpeed = FMath::Clamp(CurrMotorSpeed, -MaxSpeed, MaxSpeed);
}


// Called every frame
void UPoleController::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

